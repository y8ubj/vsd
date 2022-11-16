# Efficient Discovery and Effective Evaluation of Visual Similarities: A Benchmark and Beyond 

The following folder contains the relevant data structures and code for evaluating DINO and Argus models on the VSD task.

---
 Data Preperation
---
The DeepFashion dataset can be downloaded from [here](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html).

The In-shop Clothes Retrieval Benchmark and Consumer-to-shop Clothes Retrieval Benchmark should be downloaded and extracted to ```datasets/img```. There should be six folders in ```datasets/img``` after extraction:
```
datasets/img/CLOTHING - Consumer-to-shop Clothes Retrieval Benchmark
datasets/img/DRESSES - Consumer-to-shop Clothes Retrieval Benchmark
datasets/img/TOPS - Consumer-to-shop Clothes Retrieval Benchmark
datasets/img/TROUSERS - Consumer-to-shop Clothes Retrieval Benchmark
datasets/img/MEN - In-shop Clothes Retrieval Benchmark
datasets/img/WOMEN - In-shop Clothes Retrieval Benchmark
```


---
 Data
---
All the datasets share the following format:
- A root dir with all the images and a mapping json file
- A dedicated ```Config``` class that contains the relevant attributes for the dataset

All the relevant datasets should be located located at: ```datasets/```, the supported datasets are:
```
['in_fashion', 'in_fashion_outshop'] 
```
The metadata json format is:
```
{
    'images': [    
                    {
                        'id': <image_id> (str) (mandatory),
                        'path': <image_relative_path> (str) (mandatory),
                        'phase': <image_train_test_fold_type> (str) (mandatory),
                        
                        'bbox': <[x, y, h, w]> (list of ints) (optional),
                        'category': <object_cvategory> (str) (optional),
                        'color': <object_color> (str) (optional),
                    },
                ...
                
                ]
}
```
The metadata json contains a dict with a single mandatory key ```images```, its value is a list of dicts.
Each dict is an item - an image with different attributes. The different ```Config``` classes located at
```datasets/configs```

---
```./datasets/image_dataset``` contains the relevant classes and objects for initializing datasets and dataloaders
following  the described data format. The main object is ```MultiTaskDatasetFactory``` that gets a ```Config```
class as an input. Few important points regarding the data generation process:

- There is a built-in mechanism for dropping product ids with few views - ```id_freq_threshold```. Product ids with less
than ```id_freq_threshold``` images will be dropped from the dataset.


Data parameters:
- ```dataset``` : (str) the dataset name
- ```metadata_field``` : (str) a name of field from the metadata that will be used for filtering results 
  (during evaluation only)
- ```in_memory``` : (bool) (default : False) if set, loaded all the images into the machine's memory (RAM). Not 
  recommended and don't have any advantage over using parallelism in the DataLoader object.
- ```id_freq_threshold``` : (int) (default : 1) minimal number of views per product ID, product IDs with less than 
  id_freq_threshold images will be dropped from the dataset.
- ```crop_bbox``` : (bool) (default : False) if set, crop the images by a certain 'bbox' as specified in the metadata
  json file.
- ```use_all_data``` : (bool) (default : True) if set, use all the data for training (the train dataset will contain
  train and test images). Since VER is used in a closed-catalog scenario we mostly train it over all the data.


---
Evaluation and metrics
---

Relevant metrics to monitor and validate:
  

- *Image based ground-truth metrics* : we used human annotators for manually tag ground truth labels for selected 
  (seed, candidate) pairs. We produce (seed, candidate) from the topK predictions of a model. The pairs are given to human annotators for manual scoring of 0/1. The tagged pairs are presented in the attached zip and should be extracted into:
  ```datasets/gt_tagging/```
  
  The ground truth label jsons have the following format:
  ```
  [
      {"key": [<seed_image_name>, <candidate_image_name>], "value": <score>},
      {"key": [<seed_image_name>, <candidate_image_name>], "value": <score>},
      {"key": [<seed_image_name>, <candidate_image_name>], "value": <score>},
      ....

  ]
  ```

---
Scripts
---

- ```eval_closed_catalog.py``` : for evaluating DINO/Argus models on the closed catalog setting, supports different models (not limited to DINO).

- ```eval_wild.py``` : for evaluating DINO/Argus models on the image in the wild setting, supports different models (not limited to DINO).
