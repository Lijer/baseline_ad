from util import dataLoading


data_list = ["apascal", "bank-additional-full_normalised",'lung-1vs5', "probe",'secom',"u2r",'ad','census','creditcard']
for index in range(6,9):
    data_path = "data/{}.csv".format(data_list[index])
    logfile = 'log/{}.log'.format(data_list[index])
    x_ori, labels_ori = dataLoading(data_path, logfile)
    print("*********{}********".format(data_list[index]))
    print(x_ori.shape)