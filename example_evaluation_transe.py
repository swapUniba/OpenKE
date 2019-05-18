import config
import models
import time

'''step 1: init operations'''
start_time = time.time()
path = "OpenKE-master/benchmarks/lastfm_OK/"
con = config.Config()
con.set_in_path(path)
con.set_import_files("OpenKE-master/res/model.vec.tf")

'''step 2: specify the evaluation task to perform (link prediction and/or triple classification)'''
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)  
con.set_work_threads(8)
con.set_dimension(64)

'''step 3: set the model and test it'''
con.init()
con.set_model(models.TransE)
con.test()

'''step 4: store the evaluation time'''
elapsed_time = time.time() - start_time
with open(path+"TIME_evaluation.txt", "a+") as f:
    f.write("Number of seconds elapsed for evaluation: " + str(elapsed_time) + "\n")