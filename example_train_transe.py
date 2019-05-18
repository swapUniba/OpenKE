import config
import models
import time

'''step 1'''
con = config.Config()
start_time = time.time()
path = "OpenKE-master/benchmarks/lastfm/"
con.set_in_path(path)

'''step 2'''
con.set_work_threads(8)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_dimension(64)
con.set_bern(0)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_train_times(1000)
con.set_opt_method("SGD")

'''step 3'''
con.set_export_files("OpenKE-master/res/model.vec.tf")
con.set_out_files("OpenKE-master/res/embedding.vec.json")

'''step 4'''
con.init()
con.set_model(models.TransE)
con.run()

'''step 5'''
elapsed_time = time.time() - start_time
with open(path+"TIME.txt", "a+") as f:
    f.write("Number of seconds elapsed for training: " + str(elapsed_time) + "\n")