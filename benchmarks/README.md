#Dataset

This directory contains the following datasets:
* LastFm 
* LibraryThing
* Movielens
* LastFm + LOD
* LibraryThing + LOD
* Movielens + LOD

All the datasets contain recommendation data. All the triples contained in these datasets are in the form *(user_id item_id rel)* which means that there is a relation rel (such as like or dislike) between user_id and item_id. The datasets wihtout LOD contain the following files:
* ** entity2id.txt**: all entities and corresponding ids, one per line. The first line is the number of entities. Note that entity2id.txt ids have to start from zero and have to be continuous (0,1,2,...). If your own dataset doesn't respect this, create a mapping file to keep track of the orignal ids respect to the new ones.
* **relation2id.txt**: all relations and corresponding ids, one per line. The first line is the number of relations.
* **train2id.txt**: training file, the first line is the number of triples for training. Then the following lines are all in the format *(head, tail, rel)* which indicates there is a relation *rel* between *head*  and *tail* . Note that train2id.txt contains ids from entitiy2id.txt and relation2id.txt instead of the names of the entities and relations. If you use your own datasets, please check the format of your training file. Files in the wrong format may cause segmentation fault.
* **valid2id.txt**: validating file, the first line is the number of triples for validating. Then the following lines are all in the format *(head, tail, rel)* .
* **test2id.txt**: testing file, the first line is the number of triples for testing. Then the following lines are all in the format *(head, tail, rel)* .
* ** type_constrain.txt**: the first line is the number of relations; the following lines are type constraints for each relation. For example, the line “1200 4 3123 1034 58 5733” means that the relation with id 1200 has 4 types of head entities (if another line with relation id equals to 1200 is written, it will refer to tail entities constraints), which are 3123, 1034, 58 and 5733.
* **mapping_items.txt**: The original datasets didn't contain continuous ids for both items and users; for this reason, this file has been created, which keep track of the original items ids (2nd column in the file) respect to the new items ids (1st column in the file) which are used in all the file mentioned.
* **mapping_users.txt**: this file contains the same logic of *mapping_items.txt* but is for the users.
* **n-n.py**: you can run this file to generate additional files i.e. * type_constrain.txt, 1-1txt, 1-n.txt, n-1.txt, n-n.txt and test2id_all.txt:*
* **1-1txt, 1-n.txt, n-1.txt, n-n.txt, test2id_all.txt**:  these files are generated from *n-n.py* .

The dataset with LOD attached, contain the same data contained in the datasets without LOD, plus additional information about items (the Linked Open Data). These new information is stored as triples in the form *(item_id property_id rel)* which means that the item item_id is linked by the relation rel (e.g. rdf_type) with the LOD property property_id (e.g. owl:Thing). The datasets with LOD contain the same file listed above, plus the additional files:
* **lodMappings.txt**: links the original LOD property names (stored in utf-8 in the 2nd column) with the LOD property names used in the dataset (1st column);
* **relation_mappings.txt**: contains the same logic of *lodMappings.txt*, but is for relations.







