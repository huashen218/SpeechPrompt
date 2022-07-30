
### Summary of related files (prompt_linear_verbalizer):
1. `SpeechPrompt/ulm_prompt/train_verbalizer.py`                 
2. `SpeechPrompt/ulm_prompt/sample_beam_verbalizer.py      `    
3. `SpeechPrompt/ulm_prompt/prompt_lm_module_verbalizer/   `  
4. `SpeechPrompt/ulm_prompt/running_scripts       `       
5. `SpeechPrompt/ulm_prompt/verbalizer_analysis/   `            


For more details of modification:

**1. SpeechPrompt/ulm_prompt/train_verbalizer.py**

- changed input_args with `"--user-dir=./prompt_lm_module_verbalizer"`;
- make `prompt_linear_verbalizer` parameters trainable;
```
    ## make linear_verbalizer params trainable ###
    for param in model.decoder.prompt_linear_verbalizer.parameters():
        param.requires_grad = True
```


**2. SpeechPrompt/ulm_prompt/sample_beam_verbalizer.py** 
- changed input_args with `"--user-dir=./prompt_lm_module_verbalizer"`;


**3. SpeechPrompt/ulm_prompt/prompt_lm_module_verbalizer/**
- add `prompt_linear_verbalizer` in the `prompt_lm.py` file:

```
    def __init__():
        ###### Linear Verbalizer ######
        self.prompt_verbalizer_dropout = Dropout(p=0.2)
        self.prompt_linear_verbalizer = torch.nn.Linear(104, 104, bias=False)
        self.temperature = 1
```

```

    def output_layer(self, features): 
        """Project features to  vocabulary size."""
        if self.adaptive_softmax is None:

            # project back to size of vocabulary
            output_projection = self.output_projection(features)

            # add prompt_linear_verbalizer with dropout and temperature 
            out = self.prompt_linear_verbalizer(output_projection)
            out = self.prompt_verbalizer_dropout(out)
            out = out / self.temperature

            return out

        else:
            return features

```

**4. SpeechPrompt/ulm_prompt/running_scripts**                      
- running scripts to start training  & inference


**5. SpeechPrompt/ulm_prompt/verbalizer_analysis/**
- some messy scripts for verbalizer analysis (not used for training / inference)