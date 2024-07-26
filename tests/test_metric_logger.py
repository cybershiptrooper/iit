from iit.utils.metric import MetricStore, MetricStoreCollection, MetricType
from iit.model_pairs.iit_model_pair import IITModelPair
from iit.model_pairs.ioi_model_pair import IOI_ModelPair

from .test_model_pairs import get_test_model_pair_ingredients


def test_metric_collection():
    mc = MetricStoreCollection(
        [MetricStore("acc", MetricType.ACCURACY), MetricStore("loss", MetricType.LOSS)]
    )
    mc.create_metric_store("new_acc", MetricType.ACCURACY)
    mc.update({"acc": 0.5, "loss": 0.2, "new_acc": 0.6})
    mc.update({"acc": 0.7, "loss": 0.1, "new_acc": 0.8})
    str_arr = [str(metric) for metric in mc.metrics]
    assert str_arr == ["acc: 60.00%", "loss: 0.1500", "new_acc: 70.00%"]
    assert mc.metrics[0].get_value() == ((0.5 + 0.7) / 2) * 100
    assert mc.metrics[1].get_value() == (0.2 + 0.1) / 2
    assert mc.metrics[2].get_value() == ((0.6 + 0.8) / 2) * 100


def test_early_stop():
    mc = MetricStoreCollection(
        [MetricStore("acc", MetricType.ACCURACY), MetricStore("loss", MetricType.LOSS)]
    )
    mc.create_metric_store("new_acc", MetricType.ACCURACY)
    mc.update({"acc": 0.5, "loss": 0.2, "new_acc": 0.6})
    mc.update({"acc": 0.7, "loss": 0.1, "new_acc": 0.8})

    ll_model, hl_model, corr, _, _ = get_test_model_pair_ingredients()
    mod_pair = IITModelPair(hl_model, ll_model, corr)

    es_condition = mod_pair._check_early_stop_condition(mc.metrics)
    assert es_condition == False
    mc.update({"acc": 0.991, "loss": 0.1, "new_acc": 0.99})
    es_condition = mod_pair._check_early_stop_condition(mc.metrics)
    assert es_condition == False
    mc = MetricStoreCollection(
        [
            MetricStore("acc", MetricType.ACCURACY),
            MetricStore("loss", MetricType.LOSS),
            MetricStore("new_acc", MetricType.ACCURACY),
        ]
    )
    mc.update({"acc": 100, "loss": 0.1, "new_acc": 100})
    es_condition = mod_pair._check_early_stop_condition(mc.metrics)
    assert es_condition == True


def test_IOI_early_stop():
    per_token_accuracy = [
        1.0, # 0
        1.0, # 1
        0.005, # 2
        0.985, # 3
        0.022, # 4
        0.019, # 5
        1.0, # 6
        1.0, # 7
        0.361, # 8
        1.0, # 9
        0.084, # 10
        0.688, # 11
        1.0, # 12
        0.332, # 13
        1.0, # 14
        1.0, # 15
    ]

    IIA = 1
    accuracy = 0.6
    strict_accuracy = 1

    mc = IOI_ModelPair.make_test_metrics()
    mc.update(
        {
            "val/iit_loss": 0.2,
            "val/IIA": IIA,
            "val/accuracy": accuracy,
            "val/strict_accuracy": strict_accuracy,
            "val/per_token_accuracy": per_token_accuracy,
        }
    )

    es_condition = IOI_ModelPair._check_early_stop_fn(mc.metrics, non_ioi_thresh=0.9)

    assert es_condition == False

    IIA = 1
    accuracy = 0.6
    strict_accuracy = 1
    
    mc = IOI_ModelPair.make_test_metrics()
    mc.update(
        {
            "val/iit_loss": 0.2,
            "val/IIA": IIA,
            "val/accuracy": accuracy,
            "val/strict_accuracy": strict_accuracy,
            "val/per_token_accuracy": per_token_accuracy,
        }
    )
    es_condition = IOI_ModelPair._check_early_stop_fn(mc.metrics, non_ioi_thresh=0.5)

    assert es_condition == True
