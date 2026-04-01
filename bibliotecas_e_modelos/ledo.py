import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


def XGBoost(set_df, features, target, lambida, gama, min_split, min_leaf, cicles, rate):

  work_df = set_df[features].copy()
  work_df["target"] = set_df[target]

  work_df["hessian"] = 2
  work_df["pred_y"] = work_df["target"].mean()
  work_df["gradient"] = (-2)*(work_df["target"] - work_df["pred_y"])
  SQRES = abs(work_df["pred_y"] - work_df["target"]).mean()
  model = list()
  SQRES_list = [SQRES]
  for i in range(0, cicles):

    tree = node_function(work_df, features, gama, lambida, min_split, min_leaf)
    model.append(tree)
    weights = list()
    for s in work_df.index:
      weights.append(predict_func(tree, work_df.loc[s]))

    weights = np.array(weights)
    work_df["pred_y"] = work_df["pred_y"] + rate*weights
    work_df["gradient"] = (-2)*(work_df["target"] - work_df["pred_y"])
    SQRES_list.append(abs(work_df["pred_y"] - work_df["target"]).mean())

  return model, SQRES_list, work_df["target"].mean()


def node_function(data, features, gama, lambida, min_split, min_leaf):

    G = data["gradient"].sum()
    H = data["hessian"].sum()

    if len(data) <= min_split:
        return {"type": "leaf", "target_mean": data["target"].mean(), "weight": (-G)/(H + lambida)}

    top_feature = None
    top_limiar = None
    gain = -float("inf")
    K = int(len(features)*0.7)
    sorted = np.random.choice(features, size = K, replace = False)
    for i in sorted:
        temp_limiars = data[i].unique()
        if len(temp_limiars) <=20:
            np.sort(temp_limiars)
            limiars =  [(temp_limiars[s] + temp_limiars[s+1])/2 for s in range(0,len(temp_limiars)-1)]
            
        else:
            limiars = np.linspace(data[i].min(), data[i].max(), 20)

        for j in limiars:
            left_mask = data[i] <= j
            right_mask = data[i] > j
            left = data[left_mask]
            right = data[right_mask]

            if len(left) < min_leaf or len(right) < min_leaf:
                continue

            Gl = left["gradient"].sum()
            Hl = left["hessian"].sum()
            Gr = right["gradient"].sum()
            Hr = right["hessian"].sum()

            temp_gain = (((Gl**2)/(Hl + lambida)) + ((Gr**2)/(Hr + lambida)) - ((G**2)/(H + lambida)))

            if temp_gain > gain:
                gain = temp_gain
                top_feature = i
                top_limiar = j

    if gain <= gama:
        return {"type": "leaf", "target_mean": data["target"].mean(), "weight": (-G)/(H + lambida)}

    left = data[data[top_feature] <= top_limiar]
    right = data[data[top_feature] > top_limiar]

    left_child = node_function(left, features, gama, lambida, min_split, min_leaf)
    right_child = node_function(right, features, gama, lambida, min_split, min_leaf)

    return {"type": "node", "left_child": left_child, "right_child": right_child, "top_feature": top_feature, "treshold": top_limiar}


def predict_func(tree, sample):
  if tree["type"] == "leaf":
    return tree["weight"]

  feature = tree["top_feature"]
  treshold = tree["treshold"]

  if sample[feature] <= treshold:
    return predict_func(tree["left_child"], sample)

  if sample[feature] > treshold:
    return predict_func(tree["right_child"], sample)
  

def result_func_XGBoos(model, basescore, sample):
    prev = basescore
    for i in model:
        adjust = predict_func(i, sample)
        prev = prev + 0.2*adjust
    
    return(prev)

# função simples que estima o preço de uma casa atraves do método KNN - distância euclidiana simples 
def KNN_felipe(t_data, variables, sample, neighborhood): 
    usable_data = t_data[variables].copy()
    usable_sample = sample[variables].copy()
    
    sqr_subtraction = (usable_data - usable_sample)**2
    
    # usable_data["sqr_subtraction"] = [sqr_subtraction.iloc[i].sum() for i in sqr_subtraction.index] - cálculo abandonado por conta de demora de execução 
    usable_data["sqr_subtraction"] = sqr_subtraction.sum(axis=1) # - versão vetorizada 
    usable_data["subtraction"] = usable_data["sqr_subtraction"]**0.5
    usable_data["price"] = t_data["price"]
    usable_data.sort_values(by = "subtraction", inplace = True)
    usable_data.reset_index(inplace = True, drop = True)

    

    estimated_price = usable_data.loc[0:neighborhood -1, "price"].mean()
    

    return(estimated_price)