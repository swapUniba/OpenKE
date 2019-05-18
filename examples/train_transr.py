import config
import models
import tensorflow as tf
import numpy as np
import time

#Train TransR based on pretrained TransE results.
#++++++++++++++TransE++++++++++++++++++++

start_time = time.time()

con = config.Config()
con.set_in_path("OpenKE-master/benchmarks/lastfm_OK/")

con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_bern(0)
con.set_dimension(64)
con.set_margin(1)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

con.init()
con.set_model(models.TransE)
con.run()
parameters = con.get_parameters("numpy")

#++++++++++++++TransR++++++++++++++++++++

conR = config.Config()
#Input training files from benchmarks/FB15K/ folder.
conR.set_in_path("OpenKE-master/benchmarks/lastfm_OK/")

conR.set_work_threads(8)
conR.set_train_times(1000)
conR.set_nbatches(100)
conR.set_alpha(0.001)
conR.set_bern(0)
conR.set_dimension(64)
conR.set_margin(1)
conR.set_ent_neg_rate(1)
conR.set_rel_neg_rate(0)
conR.set_opt_method("SGD")

#Models will be exported via tf.Saver() automatically.
conR.set_export_files("OpenKE-master/res/model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
conR.set_out_files("OpenKE-master/res/embedding.vec.json")
#Initialize experimental settings.
conR.init()
#Load pretrained TransE results.
conR.set_model(models.TransR)
parameters["transfer_matrix"] = np.array([(np.identity(64).reshape((64*64))) for i in range(conR.get_rel_total())])
conR.set_parameters(parameters)
#Train the model.
conR.run()

elapsed_time = time.time() - start_time
print("Number of seconds elapsed for training: " + str(elapsed_time))



