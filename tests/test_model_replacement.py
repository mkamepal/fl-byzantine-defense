"""Model replacement arithmetic and actual eager Keras/Flower lifecycle checks."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import keras
import numpy as np
import tensorflow as tf
from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict

from fl_defense import client_app
from fl_defense.attacks.config import AttackConfig, assign_attacks
from fl_defense.attacks.datasets import prepare_attack_datasets, load_training_data
from fl_defense.attacks.model_replacement import (
    population_scale, choose_scale, replace_model, constrained_loss,
    load_assignment, weights_digest, log_round_diagnostics,
)
from fl_defense.aggregation.logged_fedavg import LoggedFedAvg
from fl_defense.data import cifar10
from fl_defense.data.partitioning import PartitionSettings
from fl_defense.detection.parameters import Parameter, model_schema
from fl_defense.utils.experiment_config import validate_experiment_config


class ReplacementUnitTests(unittest.TestCase):
    def test_shared_quota_and_old_assignments(self):
        old, old_report = assign_attacks(15, AttackConfig(enabled=True, num_malicious=7))
        cfg = AttackConfig(enabled=True, num_malicious=7,
                           enabled_types=('label_flip', 'backdoor', 'model_replacement'),
                           distribution={'label_flip': .4, 'backdoor': .3, 'model_replacement': .3})
        new, report = assign_attacks(15, cfg)
        self.assertEqual(set(old), set(new))
        self.assertEqual(report['attack_counts'], {'label_flip': 3, 'backdoor': 2, 'model_replacement': 2})
        self.assertEqual(old_report['attack_counts'], {'label_flip': 4, 'backdoor': 3})
        self.assertEqual(assign_attacks(15, cfg)[0], new)
        for n in (1, 2, 3, 6, 17):
            mapping, info = assign_attacks(n, cfg)
            self.assertEqual(sum(info['attack_counts'].values()), len(mapping))
            self.assertLessEqual(len(mapping), 7)
            self.assertNotIn('client_0', mapping)
        mapping, _ = assign_attacks(7, replace(cfg, head_client='client_1'), 'client_2')
        self.assertFalse(set(mapping) & {'client_0', 'client_1', 'client_2'})

    def test_schedule_and_invalid_configuration(self):
        cfg = AttackConfig(replacement_start_round=2, replacement_end_round=6, replacement_interval=2)
        self.assertEqual([r for r in range(1,9) if cfg.replacement_active(r)], [2,4,6])
        for kw in ({'replacement_scale': 0}, {'replacement_scale': float('nan')},
                   {'replacement_start_round': True}, {'replacement_interval': 0},
                   {'replacement_start_round': 3, 'replacement_end_round': 2},
                   {'replacement_variant': 'bad'}, {'replacement_scale_mode': 'actual_selected'},
                   {'replacement_learning_rate': -1}, {'replacement_epochs': -1},
                   {'replacement_alpha': 0}, {'replacement_alpha': 1.1}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                AttackConfig(**kw)
        with self.assertRaisesRegex(ValueError, 'start round'):
            validate_experiment_config({'num-clients': 3, 'num-server-rounds': 2,
                'attacks-enabled': True, 'attacks-enabled-types': '["model_replacement"]',
                'attacks-distribution': '{"model_replacement":1}', 'attacks-model-replacement-start-round': 3})

    def test_arithmetic_schema_overflow_and_buffers(self):
        schema = (Parameter('kernel',(2,),True), Parameter('bn/moving_variance',(2,),False))
        base = [np.array([1,2],np.float32), np.array([1,1],np.float32)]
        local = [np.array([2,0],np.float32), np.array([.2,2],np.float32)]
        attacked, d = replace_model(local, base, schema, 3)
        np.testing.assert_array_equal(attacked[0], [4,-4])
        np.testing.assert_array_equal(attacked[1], local[1])
        np.testing.assert_array_equal(base[0], [1,2])
        np.testing.assert_array_equal(local[0], [2,0])
        self.assertFalse(np.shares_memory(attacked[1],local[1]))
        self.assertAlmostEqual(d['submitted_trainable_l2'],3*d['pre_scale_trainable_l2'])
        for bad in (local[:1], [local[0].astype(np.float64),local[1]], [np.array([np.nan,0],np.float32),local[1]]):
            with self.assertRaises(ValueError): replace_model(bad,base,schema,2)
        with self.assertRaises(ValueError): replace_model(local,base,schema,1e300)
        same, zero = replace_model(base,base,schema,10)
        self.assertTrue(zero['zero_update'])
        np.testing.assert_array_equal(same[0],base[0])

    def test_population_scale_counts_only_replacement_sample_weight(self):
        summary={'clients':[{'training_samples':50,'attack':None}, {'training_samples':20,'attack':'label_flip'},
                            {'training_samples':20,'attack':'model_replacement'}, {'training_samples':10,'attack':'model_replacement'}]}
        self.assertAlmostEqual(population_scale(summary)['estimated_scale'],100/30)
        self.assertEqual(choose_scale(AttackConfig(replacement_scale_mode='fixed',replacement_scale=4),summary),4)
        self.assertAlmostEqual(choose_scale(AttackConfig(),summary),100/30)
        for c in summary['clients']:c['attack']=None
        self.assertIsNone(population_scale(summary)['estimated_scale'])
        with self.assertRaises(ValueError):choose_scale(AttackConfig(),summary)

    def test_constrained_loss_gradients_and_restoration(self):
        model=keras.Sequential([keras.Input((1,)),keras.layers.Dense(1,use_bias=False,kernel_regularizer=keras.regularizers.l2(.1))])
        model.compile(optimizer='sgd',loss='mse',run_eagerly=True)
        model.set_weights([np.array([[2]],np.float32)])
        reference=[np.zeros((1,1),np.float32)]
        x=tf.ones((1,1)); y=tf.zeros((1,1))
        with constrained_loss(model,reference,model_schema(model),.25):
            with tf.GradientTape() as tape:
                loss=model.compute_loss(x=x,y=y,y_pred=model(x),training=True)
            # .25*4 + .75*4 + .1*4 = 4.4; derivative = 4.4
            self.assertAlmostEqual(float(loss),4.4,places=5)
            self.assertAlmostEqual(float(tape.gradient(loss,model.trainable_weights)[0][0,0]),4.4,places=5)
        self.assertNotIn('compute_loss',model.__dict__)
        with self.assertRaisesRegex(RuntimeError,'test'):
            with constrained_loss(model,reference,model_schema(model),.5):raise RuntimeError('test')
        self.assertNotIn('compute_loss',model.__dict__)
        keras.backend.clear_session()


def tiny_model(learning_rate=.01):
    model=keras.Sequential([keras.Input((32,32,3)),keras.layers.GlobalAveragePooling2D(),
                            keras.layers.BatchNormalization(),keras.layers.Dense(10,activation='softmax')])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'],jit_compile=False)
    return model


class ReplacementLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        cifar10._partition_plan.cache_clear();self.addCleanup(cifar10._partition_plan.cache_clear)
        raw=np.random.RandomState(8).randint(0,200,(120,32,32,3),dtype=np.uint8)
        labels=np.arange(120)%10
        fixture=patch.object(cifar10,'_load_local_training_data',return_value=(raw,labels));fixture.start();self.addCleanup(fixture.stop)
        preview=patch('fl_defense.utils.image_previews.log_labeled_previews');preview.start();self.addCleanup(preview.stop)
        self.config={'num-clients':3,'num-server-rounds':3,'dataset-dir':str(self.root), 'local-epochs':1,
                     'batch-size':16,'learning-rate':.01,'partition-method':'iid','clustering-enabled':False,
                     'attacks-enabled':True,'attacks-num-malicious':1,'attacks-enabled-types':'["model_replacement"]',
                     'attacks-distribution':'{"model_replacement":1}', 'attacks-model-replacement-scale-mode':'fixed',
                     'attacks-model-replacement-scale':2., 'attacks-model-replacement-start-round':2,
                     'attacks-model-replacement-end-round':2,'verbose':False}
        self.logger=Mock(run_dir=self.root)
        self.overlays=prepare_attack_datasets(self.logger,self.config)
        self.manifest=json.loads((self.root/'attacks/assignment_and_poisoning.json').read_text())
        self.cid=next(iter(self.manifest['assignments']));self.partition=int(self.cid.split('_')[1])
        self.addCleanup(keras.backend.clear_session)

    def request(self, rnd, partition=None, arrays=None):
        part=self.partition if partition is None else partition
        if arrays is None:arrays=ArrayRecord(tiny_model().get_weights())
        message=Message(RecordDict({'arrays':arrays,'config':ConfigRecord({'server-round':rnd,'poisoning-data-dir':self.overlays})}),dst_node_id=100+part,message_type='train')
        context=SimpleNamespace(run_config=self.config,node_config={'partition-id':part,'num-partitions':3})
        return message,context

    def test_fixed_overlays_inactive_clean_and_manifest_mismatch(self):
        clean=cifar10.load_client_split(self.partition,3,'train',self.root,PartitionSettings())
        inactive=load_training_data(self.partition,3,self.root,PartitionSettings(),self.overlays,apply_overlay=False)
        active=load_training_data(self.partition,3,self.root,PartitionSettings(),self.overlays)
        for a,b in zip(clean,inactive,strict=True):np.testing.assert_array_equal(a,b)
        self.assertGreater(np.count_nonzero(active[0]!=clean[0]),0)
        second=load_training_data(self.partition,3,self.root,PartitionSettings(),self.overlays)
        for a,b in zip(active,second,strict=True):np.testing.assert_array_equal(a,b)
        settings=AttackConfig.from_config(self.config)
        load_assignment(self.overlays,settings,3,'client_0')
        with self.assertRaisesRegex(ValueError,'manifest'):
            load_assignment(self.overlays,replace(settings,replacement_scale=3),3,'client_0')

    def test_actual_training_scheduled_scaling_and_no_reply_oracle(self):
        base=tiny_model().get_weights()
        with patch.object(client_app,'load_model',side_effect=tiny_model), \
             patch.object(client_app,'replace_model',wraps=replace_model) as transform:
            for rnd in (1,2,3):
                msg,context=self.request(rnd,arrays=ArrayRecord(base))
                reply=client_app.train(msg,context)
                self.assertEqual(set(reply.content),{'arrays','metrics','client-info'})
                self.assertNotIn('attack',reply.content['client-info'])
                data=json.loads((self.root/f'attacks/model_replacement/round_{rnd:03d}/{self.cid}.json').read_text())
                self.assertEqual(data['active'],rnd==2)
                self.assertEqual(data['scale'],2 if rnd==2 else 1)
                self.assertEqual(data['submitted_weights_sha256'],weights_digest(reply.content['arrays'].to_numpy_ndarrays()))
                if rnd==2:
                    local,reference,schema,scale=transform.call_args.args
                    expected,_=replace_model(local,reference,schema,scale)
                    for a,b in zip(expected,reply.content['arrays'].to_numpy_ndarrays(),strict=True):np.testing.assert_array_equal(a,b)
                log_round_diagnostics(self.logger,self.overlays,rnd,[reply],[])
                logged=json.loads((self.root/f'attacks/model_replacement/round_{rnd:03d}.json').read_text())
                self.assertFalse(logged['clients'][0]['selected'])
                self.assertEqual(logged['clients'][0]['submission_state'],'received')
            self.assertEqual(transform.call_count,1)

    def test_constrain_variant_real_fit_and_asr_without_backdoor_type(self):
        # Rebuild artifacts in a separate run directory with the effective variant.
        newroot=self.root/'constrained';newroot.mkdir();self.logger=Mock(run_dir=newroot)
        self.config['attacks-model-replacement-variant']='constrain_and_scale'
        self.config['attacks-model-replacement-epochs']=2
        self.config['attacks-model-replacement-learning-rate']=.02
        self.overlays=prepare_attack_datasets(self.logger,self.config)
        msg,context=self.request(2)
        with patch.object(client_app,'load_model',side_effect=tiny_model) as factory:
            reply=client_app.train(msg,context)
            self.assertEqual(factory.call_args.args,(.02,))
            self.assertEqual(reply.content['client-info']['local-epochs'],2)
            evaluation=Message(RecordDict({'arrays':reply.content['arrays'],'config':ConfigRecord({'server-round':2})}),dst_node_id=100+self.partition,message_type='evaluate')
            result=client_app.evaluate(evaluation,context)
            self.assertIn('backdoor-asr',result.content)
            self.assertGreater(result.content['backdoor-asr']['eligible'],0)

    def test_sample_weighted_fedavg_uses_actual_submissions(self):
        strategy=LoggedFedAvg(self.logger,3,poisoning_dir=self.overlays)
        arrays=ArrayRecord(tiny_model().get_weights())
        grid=SimpleNamespace(get_node_ids=lambda:[100,101,102])
        requests=strategy.configure_train(2,arrays,ConfigRecord(),grid)
        replies=[]
        with patch.object(client_app,'load_model',side_effect=tiny_model):
            for request in requests:
                context=SimpleNamespace(run_config=self.config,node_config={'partition-id':request.metadata.dst_node_id-100,'num-partitions':3})
                replies.append(client_app.train(request,context))
        total=sum(r.content['metrics']['num-examples'] for r in replies)
        expected=[sum(r.content['arrays'].to_numpy_ndarrays()[i]*r.content['metrics']['num-examples']/total for r in replies) for i in range(len(arrays))]
        aggregate,_=strategy.aggregate_train(2,replies)
        for a,b in zip(expected,aggregate.to_numpy_ndarrays(),strict=True):np.testing.assert_allclose(a,b,atol=1e-6)
        report=json.loads((self.root/'attacks/model_replacement/round_002.json').read_text())
        self.assertTrue(report['clients'][0]['selected'])
        self.assertAlmostEqual(report['clients'][0]['accepted_weight'],1/3)


if __name__=='__main__':unittest.main()
