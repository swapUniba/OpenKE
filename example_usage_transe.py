import config
import models

'''step 1: Load the dataset and embedding files'''
con = config.Config()
con.set_in_path("OpenKE-master/benchmarks/lastfm_OK/")
con.set_import_files("OpenKE-master/res/model.vec.tf")    
con.set_work_threads(8)
con.set_dimension(64)

'''step 2: init the model'''
con.init()
con.set_model(models.TransE)

'''step 3: perform your operation'''
con.predict_head_entity(1928, 1, 5)
con.predict_tail_entity(0, 1, 5)
con.predict_relation(0, 1928, 5)
con.predict_triple(0, 1928, 1)