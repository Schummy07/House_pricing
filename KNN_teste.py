import pandas as pd 
import numpy as np 

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