import pandas as pd
from aitextgen.TokenDataset import TokenDataset, TokenDatasetList
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
import os
from pkg_resources import resource_filename
import gzip
import pickle as pickle
from torch.utils.data import DataLoader

# create datasets (assume already scraped via terminal)

# find list of work IDs

metadata = pd.read_csv('fanfics_metadata.csv')
print(metadata.keys())
ids = metadata["work_id"]
print(ids)

# create list of TokenDatasets, one for each fic
stories = []
count = 0
total_tokens = 0

STATIC_PATH = resource_filename(__name__, "aitextgen/static")

tokenizer = GPT2TokenizerFast(vocab_file=os.path.join(STATIC_PATH, "gpt2_vocab.json"), merges_file = os.path.join(STATIC_PATH, "gpt2_merges.txt"), padding_side="left")
                # https://github.com/huggingface/transformers/issues/10202
tokenizer.add_special_tokens({"additional_special_tokens": ["<|endoftext|>"]})

print(tokenizer.pad_token)

with gzip.open('dataset_list_cache.p', 'rb') as inp:
    full_dataset = pickle.load(inp)

dataloader = DataLoader(
            full_dataset,
            8,
            shuffle=True,
        )

data_iter = (dataloader._get_iterator())

#for input in data_iter:
    #print(input)




for i in range(len(ids)):
    file_name = "fanfics.csv_text_files/"+str(ids[i])+".txt"
    try:
        new_story = TokenDataset(file_name, tokenizer=tokenizer, padding_side="left")
        stories.append(new_story)
        print("story token length: ",len(new_story.tokens))
        count += 1
        total_tokens += len(new_story.tokens)
    except(BaseException):
        print("Failed")
        pass


print(count)
print("total tokens: ",total_tokens)
#print(tokenizer.decode(stories[0].tokens))



# merge all TokenDatasets
dataset_list = TokenDatasetList(stories)
print("starting save!")
print(dataset_list.full_length)
dataset_list.save()

exit()
#full_dataset = aitextgen.TokenDataset.merge_datasets(stories, equalize=False)
#print(len(full_dataset.tokens))
#decoded = tokenizer.decode(full_dataset.tokens)
#print(len(decoded))
#print(decoded)
#full_dataset.save()
