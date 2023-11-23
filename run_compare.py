import hashlib
import json
import os

from recbole.quick_start import run_recbole
import yaml

from recbole.utils import get_local_time, ensure_dir

COMPARE_ROOT = "./compared"


def write_finished(finished, finished_log_path):
    """
    写入已完成的列表
    """
    with open(finished_log_path, mode="w+", encoding="utf8") as f:
        json.dump(finished, f)


def load_finished(finished_log_path):
    """
    读取已完成列表
    """
    if not os.path.exists(finished_log_path):
        return {}
    with open(finished_log_path, mode="r", encoding="utf8") as f:
        return json.load(f)


if __name__ == '__main__':
    ensure_dir(os.path.dirname(COMPARE_ROOT))
    # 0. 加载配置
    with open("compared.yml", mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    compare_dir = config['dir'] if 'dir' in config else "compared"
    ensure_dir(os.path.join(COMPARE_ROOT, compare_dir))
    # 已完成列表数据的加载
    finished_log_path = os.path.join(COMPARE_ROOT, "{}/finished.json".format(compare_dir))
    finished = load_finished(finished_log_path)
    # 1. 遍历所有对比任务进行训练
    datasets = config['datasets']
    for dataset in datasets:
        models = config['datasets'][dataset]['models']
        args = config['datasets'][dataset]
        del args['models']
        for model in models:
            if isinstance(model, str):
                model_name = model
            else:
                model_name = list(model.keys())[0]
                for k in model[model_name]:
                    args[k] = model[model_name][k]
            if dataset in finished and model_name in finished[dataset] and finished[dataset][model_name]:
                continue
            config_file_list = []
            if os.path.exists("./properties/dataset/{}.yaml".format(dataset)):
                config_file_list.append("./properties/dataset/{}.yaml".format(dataset))
            if os.path.exists("./properties/model/{}.yaml".format(model_name)):
                config_file_list.append("./properties/model/{}.yaml".format(model_name))
            config_file_list.append("./properties/overall.yaml")
            # 2. 训练，采用默认配置
            metrics = run_recbole(
                model=model_name, dataset=dataset, config_dict=args, config_file_list=config_file_list
            )
            # 3. 保存数据
            temp = "{}/{}-{}-temp.json".format(
                compare_dir, model_name, dataset
            )
            compare_temp_path = os.path.join(COMPARE_ROOT, temp)
            # 如果已存在则删除，因为并没有完成
            if os.path.exists(compare_temp_path):
                os.remove(compare_temp_path)
            with open(compare_temp_path, mode="w+", encoding="utf8") as f:
                json.dump(metrics, f)
            if dataset not in finished:
                finished[dataset] = {}
            finished[dataset][model_name] = 1
            write_finished(finished, finished_log_path)
    # 4. 合并结果
    for dataset in finished:
        models = finished[dataset]
        for model_name in models:
            temp = "{}/{}-{}-temp.json".format(
                compare_dir, model_name, dataset
            )
            compare_temp_path = os.path.join(COMPARE_ROOT, temp)
            with open(compare_temp_path, mode="r", encoding="utf8") as f:
                finished[dataset][model_name] = json.load(f)
    result_name = "{}/result.json".format(
        compare_dir
    )
    compare_temp_path = os.path.join(COMPARE_ROOT, result_name)
    with open(compare_temp_path, mode="w+", encoding="utf8") as f:
        json.dump(finished, f)
    # 5. 清除
    for dataset in finished:
        models = finished[dataset]
        for model_name in models:
            temp = "{}/{}-{}-{}-temp.json".format(
                compare_dir, model_name, dataset, get_local_time()
            )
            compare_temp_path = os.path.join(COMPARE_ROOT, temp)
            if os.path.exists(compare_temp_path):
                os.remove(compare_temp_path)