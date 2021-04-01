#%%
 
from utils import *

max_seq_length=32 ## ensure of the form 2^(odd integer)
stride=10 ## (1/3 * )
assert ((max_seq_length-2)%stride==0)

# chunk_to_len()
# %%
