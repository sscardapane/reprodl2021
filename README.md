# Reproducible Deep Learning
## PhD Course in Data Science, 2021, 3 CFU
[[Official website](https://www.sscardapane.it/teaching/reproducibledl/)]

### Adversarial Attack
On the https://landscape.lfai.foundation/ in the Trusted and Responsible AI (AI Explainability 360 Toolkit), we can find an Adversarial Attack framework. 
In particular, I selected the Adversarial Robustness Toolbox (ART) by IBM, and I applied it to the audionet classificaton model.
In the notebook, I show how to create adversarial examples of audio data with ART. 
In the visualization part, it is possible to compare the original and the perturbed instance.

In the model/ folder, you can find the audionet model already trained.

## Prerequisites 

Install requirements: 
- Install the ART toolbox 
```bash
pip install adversarial-robustness-toolbox
```
## Instructions to implement Adversariol Attack

Follow the notebook 'Adversarial Notebook.ipynb':

0. Execute the notebook until step 3.
1. Run step 3 if you want to train the model, otherwise jump to step 4 an load the model
2. Import the ART toolbox to get the perturbation function and model function:
```python
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from art import config
from art.defences.preprocessor import Mp3Compression
from art.utils import get_file
```
3. Test the perturbation for adversarial attack, 
where '''epsilon''' can be change to set the maximum perturbation
```python
epsilon = .1
pgd = ProjectedGradientDescent(classifier_art, eps=epsilon)
adv_waveform = pgd.generate(
    x=torch.unsqueeze(waveform, 0).numpy()
)
```
4. Visualize the class of the original waveform and the perturber waveform
```python
with torch.no_grad():
    _, pred = torch.max(audionet(torch.unsqueeze(waveform, 0)), 1)
    _, pred_adv = torch.max(audionet(torch.from_numpy(adv_waveform)), 1)

# print results
print(f"Original prediction (ground truth):\t{pred.tolist()[0]} ({label})")
print(f"Adversarial prediction:\t\t\t{pred_adv.tolist()[0]}")
```
5. Visualize the original waveform and the perturber waveform
```python
def display_waveform(waveform, title="", sr=8000):
    """Display waveform plot and audio play UI."""
    plt.figure()
    plt.title(title)
    plt.plot(waveform)
    ipd.display(ipd.Audio(waveform, rate=sr))
```
