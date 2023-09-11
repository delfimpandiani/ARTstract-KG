from pathlib import Path
from typing import Tuple, List, Set

from functools import cached_property

from collections import defaultdict
from more_itertools import collapse
from itertools import product, permutations
import json

import numpy as np

PATH = Path("/home/n28div/university/phd/ARTstract-KG/main/perceptual_semantics_detection/merged_data.json")

class DistributionalStatistics(object):

  def __init__(self, path: Path):
    self.data = self.__extract_dict(path)
    
  def __extract_dict(self, json_path: Path) -> Tuple[dict, dict]:
    perceptional = json.load(json_path.open())
    
    images = []
    for image_id in perceptional.keys():
      image_d = dict()
      image_d["id"] = image_id
      
      image_data = perceptional[image_id]
      # extract action
      k = next(iter(image_data["act"].keys()))
      image_d["action"] = image_data["act"][k]["conceptnet_concept"]
      
      # extract emotions
      k = next(iter(image_data["em"].keys()))
      image_d["emotion"] = image_data["em"][k]["conceptnet_concept"]

      # detected objects
      k = next(iter(image_data["od"].keys()))
      image_d["objects"] = list(set(jo["detected_object"] for jo in image_data["od"][k]["detected_objects"]))
      
      # art style
      k = next(iter(image_data["as"].keys()))
      image_d["art_style"] = image_data["as"][k]["conceptnet_concept"]
      
      # color
      k = next(iter(image_data["color"].keys()))
      image_d["objects"] = list(set(jo["conceptnet_concept"] for jo in image_data["color"][k]))
      
      # human-presence
      k = next(iter(image_data["hp"].keys()))
      image_d["human_presence"] = image_data["hp"][k]["human_presence"]
      
      # age detected
      k = next(iter(image_data["age"].keys()))
      image_d["age"] = image_data["age"][k]["conceptnet_concept"]
      
      images.append(image_d)
    return images

  @cached_property
  def perceptions(self) -> List["Perception"]:
    return [Perception(self, k) for k in self.data[0].keys() if k != "id"]

  def __iter__(self):
    return iter(self.data)

  def __len__(self) -> int:
    return len(self.data)


class Perception(object):

  def __init__(self, ds: DistributionalStatistics, label: str):
    self.__ds = ds
    self.__label = label

  def __str__(self) -> str:
    return self.__label

  def __repr__(self) -> str:
    return f"<Perception: {str(self)}>"

  @cached_property
  def values(self) -> List["PerceptualValue"]:
    values = list(set(collapse([image[self.__label] for image in self.__ds])))
    return [PerceptualValue(self.__ds, self, v) for v in values]
  

class PerceptualValue(object):
  def __init__(self, ds: DistributionalStatistics, perception: Perception, label: str):
    self.__perception = perception
    self.__ds = ds
    self.__label = label

  def __str__(self) -> str:
    return self.__label

  @cached_property
  def perception(self) -> str:
    return self.__perception

  @cached_property
  def images(self) -> Set[str]:
    contains = lambda x: (type(x) is list and str(self) in x) or (x == str(self))
    perception_str = str(self.perception)
    return set(collapse([image["id"] for image in self.__ds if contains(image[perception_str])]))

  @cached_property
  def probability(self) -> float:
    return len(self.images) / len(self.__ds)


def mutual_information(ds: DistributionalStatistics, X: Perception, Y: Perception):
  i_xy = 0
  for x, y in product(X.values, Y.values):
    joint_p = len(x.images.intersection(y.images)) / len(x.images.union(y.images))
    p_x = len(x.images) / len(ds)
    p_y = len(y.images) / len(ds)    

    # compute MI and add it to accumulation variable
    i_xy += joint_p * np.log(joint_p / (p_x * p_y))

  return i_xy


def joint_entropy(X: Perception, Y: Perception):
  h_hy = 0
  for x, y in product(X.values, Y.values):
    # joint probability
    joint_p = len(x.images.intersection(y.images)) / len(x.images.union(y.images))
    conditional_p = len(x.images.intersection(y.images)) / len(y.images)
    h_hy += joint_p * np.log(conditional_p + 1e-20)
  
  return -1 * h_hy

def entropy(X: Perception):
  return -1 * sum([v.probability * np.log(v.probability) for v in X.values])

ds = DistributionalStatistics(PATH)
images = [x["id"] for x in ds.data[:100]]

print(joint_entropy(ds.perceptions[0], ds.perceptions[4]))

# #m = np.array(entropies).reshape((len(ds.perceptions), len(ds.perceptions)))
# entropies = np.zeros((len(ds.perceptions), len(ds.perceptions)))
# for idx_a, perc_a in enumerate(ds.perceptions):
#   for idx_b, perc_b in enumerate(ds.perceptions):
#     h_a_b = joint_entropy(perc_a, perc_b)
#     h_b = entropy(perc_b)
#     entropies[idx_a, idx_b] = h_a_b - h_b


# import matplotlib.pyplot as plt

# entropies = entropies / np.diag(entropies).reshape(-1, 1)

# fig, ax = plt.subplots()
# im = ax.imshow(entropies)
# fig.colorbar(im, orientation='vertical')

# ax.set_xticklabels([""] + [str(x) for x in ds.perceptions])
# ax.set_yticklabels([""] + [str(x) for x in ds.perceptions])

# plt.show()


