import re
import math

%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns

text = 'Time travel is the concept of movement between certain points in time, analogous to movement between different points in space by an object or a person, typically using a hypothetical device known as a time machine. Time travel is a widely-recognized concept in philosophy and fiction. The idea of a time machine was popularized by H. G. Wells 1895 novel The Time Machine. It is uncertain if time travel to the past is physically possible. Forward time travel, outside the usual sense of the perception of time, is an extensively-observed phenomenon and well-understood within the framework of special relativity and general relativity. However, making one body advance or delay more than a few milliseconds compared to another body is not feasible with current technology. As for backwards time travel, it is possible to find solutions in general relativity that allow for it, but the solutions require conditions that may not be physically possible. Traveling to an arbitrary point in spacetime has a very limited support in theoretical physics, and usually only connected with quantum mechanics or wormholes, also known as Einstein-Rosen bridges.'
# To remove numbers from the text
# result = re.sub(r"\d", "", text)
text = text.lower()
arr = text.split(" ")
uniqueValues, occurCount = np.unique(np.array(arr), return_counts=True)
N = len(occurCount)

proportion = []
for i in occurCount:
    proportion.append(i/len(arr))
    
def zipfs_law(N, k, s= .65):
    return (1/math.pow(k,s))/((np.sum(1/(np.arange(1,N +1)**s))))

data = pd.DataFrame({'Word': uniqueValues, 'Count': occurCount, 'actual_proportion': proportion})
data['prediction'] = prediction
data.head()

data = data.sort_values(by = ['Count'], ascending=False).reset_index()

prediction = []
for i in range(1,N+1):
    prediction.append(zipfs_law(N, i))

se = pd.Series(prediction)
data['prediction'] = np.array(prediction)
print(data.shape)


# Visualization

sns.distplot(prediction)
sns.distplot(data['actual_proportion'])

data.drop(['Count'], axis = 1, inplace = True)
data.plot()
