import numpy as np
import pandas as pd
import torch
import json as js
import nltk
nltk.download('averaged_perceptron_tagger_eng', quiet= True)
import os
from tqdm import tqdm
from scipy import linalg
import argparse
import utils
import matplotlib.pyplot as plt
device = "auto" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("model", help="A model identifier string on https://huggingface.co", type=str)
parser.add_argument("--populations", help="A json file containing a dictionary of population terms of format {population_name: list[str], population_name: list[str]}.", type=str, default="names.json")
parser.add_argument("--examples", help="A text file with context examples for terms in the stereotype dimensions dictionary or 'None' for no context.", type=str, default="generated_examples.txt")
parser.add_argument("--hf_token", help="A user access token on https://huggingface.co to access gated models.", type=str, default=None)
parser.add_argument('--standardization', help="Whether to standardize the results over the defined populations.", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--prefix", help="A prefix for the result files.", type=str, default="")
args = parser.parse_args()


### Loading the dictionaries and specifying dimensions
stereodim_dictionary = pd.read_csv("./dictionaries/stereotype_dimensions_dictionary.csv", index_col = 0)
stereotype_dimensions = ["Sociability", "Morality", "Ability", "Agency", "Status", "Politics", "Religion"]
warmth_competence_dimensions = {"Warmth": ["Sociability", "Morality"], "Competence": ["Ability", "Agency"]}
all_dimensions = stereotype_dimensions + list(warmth_competence_dimensions.keys())

### loading the examples for embedding terms in the stereotype dimensions dictionary
if args.examples != 'None':
    max_examples = 5
    context_examples = {}
    with open(args.examples, "r") as f:
        for entry in f.readlines():
            entry = js.loads(entry)
            context_examples[f'{entry["term"]} - {entry["synset"]}'] = entry["examples"][:max_examples]
    no_context = False
else:
    no_context = True

### loading the population terms for projection
with open(args.populations, "r") as f:
    populations = js.load(f)  
group1 = list(populations.keys())[0]
group2 = list(populations.keys())[1]

### setting up the embedding model and tokenizer
tokenizer, embedding_model = utils.load_model_for_embedding_retrieval(args.model, device, hf_token=args.hf_token)
layers = [i for i in range(utils.get_number_of_hidden_states(tokenizer, embedding_model))]
model_name = args.model.split("/")[-1]
    
### creating the folder structure for layerwise saving
if not os.path.isdir("./embeddings"):
        os.mkdir("./embeddings")
for layer in layers:
    if not os.path.isdir(f"./embeddings/{model_name}-L{layer}"):
        os.mkdir(f"./embeddings/{model_name}-L{layer}")


# Retrieving sense embeddings and embeddings of stereotype dimensions

### initializing objects for storing embeddings
embedding_dict = {layer:{"sense_embeddings": [], "sense_embedding_labels": [],
                         "pole_embedding_dict": {}} for layer in layers}
    
### looping over the word senses in the stereotype dimensions dictionary
for index, row in tqdm(stereodim_dictionary.iterrows(), desc="Retrieving embeddings for stereotype dimensions."):
 
    term = row["term"]
    synset = row["synset"]
    dimension = row['dimension']
    direction = row['dir']
    
    # retrieving the contexts
    contexts = [term] if no_context else context_examples[term + " - " + synset]

    # skipping the row if there are no context examples
    if len(contexts)== 0:
        print(f"No examples for term '{term}'.")
        continue
    
    # computing the term (sense) embedding for each layer and context
    layerwise_sense_embeddings = [utils.get_word_embedding_by_layer(tokenizer, embedding_model, context, term, layers) for context in contexts]
    
    # averaging across contexts
    layerwise_sense_embeddings = torch.stack(layerwise_sense_embeddings).mean(dim=0)

    # layerwise processing of the embeddings
    for layer, layer_dict in embedding_dict.items():

        sense_embedding = layerwise_sense_embeddings[layer]

        # storing sense embeddings for their respective dimension + direction
        if dimension + "-" + direction not in layer_dict["pole_embedding_dict"].keys():
            layer_dict["pole_embedding_dict"][dimension + "-" + direction] = [] 
        layer_dict["pole_embedding_dict"][dimension + "-" + direction].append(sense_embedding)

         # storing the individual sense embedding 
        layer_dict["sense_embeddings"].append(sense_embedding)
        # storing the sense embedding label
        layer_dict["sense_embedding_labels"].append(term + " - " + synset + " - " + dimension + " - " + direction)
        
           
### layerwise saving
for layer, layer_dict in embedding_dict.items():
    
    # reformatting and saving stereotype pole embeddings
    for pole, embeddings in layer_dict["pole_embedding_dict"].items():
        embeddings = np.vstack(embeddings)
        with open(f"./embeddings/{model_name}-L{layer}/{pole}_embeddings.npy", "wb") as f:
            np.save(f, embeddings)
    
    # for high level warmth and competence dimensions
    for dim, subdimensions in warmth_competence_dimensions.items():
        for dir_ in ["low", "high"]:
            # concatenating subdimension embeddings
            embeddings = np.concatenate([layer_dict["pole_embedding_dict"][subdimensions[0] + "-" + dir_], layer_dict["pole_embedding_dict"][subdimensions[1] + "-" + dir_]], axis = 0)
            with open(f"./embeddings/{model_name}-L{layer}/{dim + '-' + dir_}_embeddings.npy", "wb") as f:
                np.save(f, embeddings)
            
    # reformatting and saving individual sense embeddings to numpy
    sense_embeddings = np.vstack(layer_dict["sense_embeddings"])
    with open(f"./embeddings/{model_name}-L{layer}/sense_embeddings.npy", "wb") as f:
            np.save(f, sense_embeddings)
    # save information on synsets and dimensions
    with open(f"./embeddings/{model_name}-L{layer}/sense_embedding_labels.txt", "w") as f:
        [f.write(dimension + "\n") for dimension in layer_dict["sense_embedding_labels"]]
        

        
# Base Change Matrices for Projection

warmth_competence_base_change_inv = {}
stereodim_base_change_inv = {}

### iterating over layers to prepare the base change matrices
for layer in tqdm(layers, desc="Preparing layerwise base change matrices."):

    # 7 stereotype dimensions
    stereodim_base_change = []

    for dim in stereotype_dimensions:
    
        with open(f"./embeddings/{model_name}-L{layer}/{dim + '-low'}_embeddings.npy", "rb") as f:
            low_pole_embeddings = np.load(f)
        with open(f"./embeddings/{model_name}-L{layer}/{dim + '-high'}_embeddings.npy", "rb") as f:
            high_pole_embeddings = np.load(f)
            
        dim_low_mean_value = np.average(low_pole_embeddings, axis = 0)
        dim_high_mean_value = np.average(high_pole_embeddings, axis = 0)
    
        # computing direction vector per stereotype dimension 
        direction_vector = dim_high_mean_value - dim_low_mean_value
            
        stereodim_base_change.append(direction_vector)

    # saving the base change matrix
    with open(f"./embeddings/{model_name}-L{layer}/stereodim_base_change.npy", "wb") as f:
        np.save(f, stereodim_base_change)

    # inverse for projection
    stereodim_base_change_inv[layer] = linalg.pinv(np.transpose(np.vstack(stereodim_base_change)))

    
    # Warmth and competence dimensions
    warmth_competence_base_change = []

    for dim in warmth_competence_dimensions.keys():

        with open(f"./embeddings/{model_name}-L{layer}/{dim + '-low'}_embeddings.npy", "rb") as f:
            low_pole_embeddings = np.load(f)
        with open(f"./embeddings/{model_name}-L{layer}/{dim + '-high'}_embeddings.npy", "rb") as f:
            high_pole_embeddings = np.load(f)
            
            dim_low_mean_value = np.average(low_pole_embeddings, axis = 0)
            dim_high_mean_value = np.average(high_pole_embeddings, axis = 0)
    
        # computing direction vector per stereotype dimension 
        direction_vector = dim_high_mean_value - dim_low_mean_value
            
        warmth_competence_base_change.append(direction_vector)

    # saving the base change matrix
    with open(f"./embeddings/{model_name}-L{layer}/warmth_competence_base_change.npy", "wb") as f:
        np.save(f, warmth_competence_base_change)

    # inverse for projection
    warmth_competence_base_change_inv[layer] = linalg.pinv(np.transpose(np.vstack(warmth_competence_base_change)))
    


# Projection of populations to stereotype dimensions

### result dictionary to hold values for warmth and competence and detailed stereotype dimensions, for all layers
result_dict = {}
for layer in layers:
    result_dict[f"{model_name}-L{layer}"] = {"Model": len(list(warmth_competence_dimensions.keys()) + stereotype_dimensions)*[f"{model_name}-L{layer}"], 
                                        "Dimension": list(warmth_competence_dimensions.keys()) + stereotype_dimensions}


### looping over the populations

for group, terms in populations.items():
    for term in tqdm(terms, desc = f"Projecting {group} to stereotype dimensions"):

        # simple templates or no context; template filling differs if the term is a proper noun, detection not always reliable with nltk.pos_tag
        isNNP = True if nltk.pos_tag([term])[0][1] == "NNP" or "names" in group.lower() else False
        contexts = [term] if no_context else [utils.fill_template(term, TEMPLATE, isNNP=isNNP) for TEMPLATE in utils.TEMPLATES]

        # computing the term embedding for each layer: retrieving the embedding for each context
        layerwise_sense_embeddings = [utils.get_word_embedding_by_layer(tokenizer, embedding_model, context, term, layers) for context in contexts]
        
        # averaging across contexts
        layerwise_sense_embeddings = torch.stack(layerwise_sense_embeddings).mean(dim=0)

        for layer in layers:
            sense_embedding = layerwise_sense_embeddings[layer]

            # projection into polar spaces
            warmth_competence_embedding = torch.matmul(torch.from_numpy(warmth_competence_base_change_inv[layer]).double(), sense_embedding.double()).numpy()
            stereodim_embedding = torch.matmul(torch.from_numpy(stereodim_base_change_inv[layer]).double(), sense_embedding.double()).numpy()
            result_dict[f"{model_name}-L{layer}"][term] = np.concatenate((warmth_competence_embedding, stereodim_embedding))    
    
results = pd.concat([pd.DataFrame.from_dict(result_dict[model_name]) for model_name in result_dict.keys()])


### averaging the results over layers for each dimension
for dimension in all_dimensions:
    new_row = [model_name] + [dimension] + list(results.loc[results["Dimension"] == dimension].set_index(["Model", "Dimension"]).mean(axis = 0))
    new_row = pd.DataFrame([new_row], columns=results.columns)
    results = pd.concat((results, new_row))

 
if len(populations.keys()) > 2:  
    print(f"More than two populations in {args.populations}, the analysis is run to compare the first two groups.")

### standardizing the results
if args.standardization == True:
    results[populations[group1] + populations[group2]] = results.apply(lambda x: utils.standardize(x[populations[group1] + populations[group2]]), axis='columns', result_type='expand')

### calculating statistics for each dimension
stats = pd.DataFrame(results.apply(lambda x: utils.get_diff(x[populations[group1]], x[populations[group2]]), axis='columns', result_type='expand'))
stats.columns = [f"{group1}_mean", f"{group1}_std", f"{group2}_mean", f"{group2}_std", "diff", "diff_pvalue", "diff_abs"]
results = pd.concat([results, stats], axis=1)


### saving the results
results.to_csv(f"./{args.prefix}{model_name}_projection_results.csv")


# Plotting the results for warmth and competence dimensions

### selection and order of stereotype dimensions
plot_dimensions = ["Competence","Warmth"]

polar_labels = {
    "Warmth": {"low": "Low Warmth", "high": "High Warmth"},
    "Competence": {"low": "Low Competence","high": "High Competence"}
         }

### setting up the plot
fig, ax1 = plt.subplots(1, 1)
    
### setting plot styles
styles = {"line": {group1: "solid", group2: "dashdot"},
            "color": {group1: "rebeccapurple", group2: "mediumorchid"}}

### dictionaries to collect the values and labels to plot
plot_values = {group1:[], group2:[]}
plot_labels = { "low labels": [], "high labels": []}          
bold_labels = [] # to make bold labels of statistically significant dimensions

### filling the dictionaries
for dimension in plot_dimensions:
    row = results.loc[(results["Dimension"] == dimension) & (results["Model"] == model_name)]
    for group in [group1, group2]:
        plot_values[group].append(row[f"{group}_mean"])

    # setting polar labels, with significance
    low_label = polar_labels[dimension]["low"]
    high_label = polar_labels[dimension]["high"]
    if row["diff_pvalue"].values[0] <0.05: # if the difference is significant
        high_label = high_label + "$^*$"
        bold_labels.append(low_label)
        bold_labels.append(high_label)
    plot_labels["low labels"].append(low_label)
    plot_labels["high labels"].append(high_label)

# plotting
for line, values in plot_values.items():
        color = styles["color"][line] if line in [group1, group2] else styles["color"]["other"]
        linestyle = styles["line"][line] if line in [group1, group2] else styles["line"]["other"]           
        ax1.plot(values, np.arange(len(values)), 
                 label = str(line), 
                 color = color, 
                 marker = "o", 
                 linestyle = linestyle)
    
### setting left side tick labels
ax1_tick_labels =  plot_labels["low labels"]
ax1.set_ylim(ax1.get_ylim()[0]-0.3, ax1.get_ylim()[1]+0.3)
ax1.set_yticks(np.arange(len(ax1_tick_labels)), labels = ax1_tick_labels, verticalalignment = "center", fontsize = 9)
for label in ax1.get_yticklabels():
    if label.get_text() in bold_labels:
        label.set_fontweight('bold')

### setting x-axis limits and label
max_x = max(map(abs, [ax1.get_xlim()[0], ax1.get_xlim()[1], 1]))
ax1.set_xlim(-max_x,max_x)
ax1.tick_params(axis='x', which='major', labelsize=8)
ax1.set_xlabel("projected values", fontsize = 8)

# setting right side tick labels
ax2_tick_labels = plot_labels["high labels"]
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.set_yticks(np.arange(len(ax2_tick_labels)), labels = ax2_tick_labels, verticalalignment = "center", fontsize = 9)
for label in ax2.get_yticklabels():
    if label.get_text() in bold_labels:
        label.set_fontweight('bold')

# setting the title and legend
ax1.set_title(model_name, fontsize = 9)
ax1.legend(bbox_to_anchor=(1, -0.45), ncol=2, fontsize = 9)

# setting the figsize and saving the plot
fig.set_figwidth(3.5)
fig.set_figheight(1.3)
plt.savefig(f'./{args.prefix}{model_name}_warmth_competence_profile.pdf', bbox_inches='tight') 

print(f"Warmth and competence profile saved in file: ./{args.prefix}{model_name}_warmth_competence_profile.pdf")








   