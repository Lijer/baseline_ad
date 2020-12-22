import os
from multiprocessing import Process


def fun1(command):
    os.system(command)


if __name__ == '__main__':
    # 多进程利用多个GPU训练模型
    process_list = []
    
    docker_command = "CUDA_VISIBLE_DEVICES={GPU} python main.py " \
                     "--data_dir ./data/{data}.csv " \
                     "--c_in {c_in} " \
                     "--h {h} " \
                     "--batch_size {batch_size}"

    # data_list = ["apascal", "bank-additional-full_normalised",  "probe","u2r","0th_ts_train",'15th_ts_train','19th_ts_train','24th_ts_train']
    data_list = ["apascal", "bank-additional-full_normalised",'lung-1vs5', "probe",'secom',"u2r",'ad','census','creditcard',
             'aid362', 'backdoor', 'celeba', 'chess', 'cmc', 'r10', 'sf', 'w7a']
    batch_size_list = [128, 256, 16, 256, 32, 256, 64, 1024, 1024, 
                128, 2048, 2048, 1024, 64, 512, 64, 1024]
    out_c_list = [50, 50, 128, 16, 64, 16, 128, 50, 15, 
                50, 64, 16, 16, 4, 32, 8, 64]
    c_in_list = [64, 62, 3312, 34, 590, 34, 1558, 511, 30, 114, 208, 39, 23, 8, 100, 19, 300]
    i = 16
    gpu_id = 3
    comand_id = [0]
    for _ in range(1):
        command = docker_command.format(GPU = gpu_id, data = data_list[i], c_in = c_in_list[i], h = out_c_list[i], batch_size = batch_size_list[i])
        p = Process(target = fun1, args=(str(command),))
        print(command)
        p.start()
        process_list.append(p)
    for i in process_list:
        p.join()
    print("测试结束")
# nohup python run.py >30_.log 2>&1 &
