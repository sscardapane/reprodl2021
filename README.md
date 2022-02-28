# Reproducible Deep Learning
## Extra: Evidently
### Author: [giu.gr](https://github.com/giu.gr)
[Evidently](https://docs.evidentlyai.com/)

Before going into this branch, please look at the main branch in order to understand the project details.
>:warning: extra branches implement additional exercises created by the students of the course to explore additional libraries and functionalities.
> They can be read independently from the main branches. Refer to the original authors for more information.

&nbsp;

This is an extra branch of **exercise1_git_completed** that is intended to introduce Evidently, a useful library to build graphical dashboards for features and performance analysis.

&nbsp;

## Goal
Integrate Evidently dashboards to improve data visualization and performance analysis. Basically Evidently is an open source Python library for machine learning engineers, that aims to be a useful tool for performance evaluation and monitoring of machine learning models, both during the development phase and in production. It can build interactive visual reports integrated with notebooks, data and model profiling dashboards, and perform real-time monitoring (further integrable with projects like [Graphana and Prometheus](https://docs.evidentlyai.com/integrations/evidently-and-grafana)).


## Step 1: installation
1. Evidently can be installed on Windows, GNU/Linux and Mac OS with PyPi:

```sh
pip install evidently
```

To integrate it with a Jupyter notebook, an nbextension is required to install:

```sh
$ jupyter nbextension install --sys-prefix --symlink --overwrite --py evidently
```
This extension can be enabled invoking the following command

```sh
$ jupyter nbextension enable evidently --py --sys-prefix
```

If you are running your project on Google Colab you do not require to install any nbextension.

2. To check your installation, you can try importing in your Python project a specific tab to plot:

```python
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import RegressionPerformanceTab
```

Check the [official documentation](https://docs.evidentlyai.com/tutorial) for further details.

&nbsp;

## Step 2: project setup and integration

Essentially, the kind of data required to Evidently to plot dashboards depends on the tabs themselves, which can be synthesized with the following categories:

* Data Drift
* Numerical Target Drift
* Categorical Target Drift
* Regression Performance
* Classification Performance
* Probabilistic Classification Performance.

Tabs we are interested in can be called through the suitable Dashboard class constructor parameters that get a list in input:

```python
test_report = Dashboard(tabs=[RegressionPerformanceTab()])
```

Next, we consider that we have validation and test data after performing processing with a given supervised learning algorithm:

```python
test_report = Dashboard(tabs=[ClassificationPerformanceTab()])

validationset_reference = {"target": val_set_labels, "prediction": val_set_predicted_labels}
testset_production = {"target": test_set_labels, "prediction": test_set_predicted_labels}

df_reference = pd.DataFrame(dict_reference)
df_production = pd.DataFrame(dict_production)
```

In the previous snippet of code we are using the validation set as reference performance and test set labels as production performance so that we can compare model behavior.

So Panda's DataFrames integration require passing classification data as a dictionary, with columns named "target" and "prediction".



## Step 3: dashboard plotting

Lastly, we can proceed processing data and showing the dashboard itself:

```python
test_report.calculate(df_reference, df_production, column_mapping = None)
test_report.show()
```

In situations where there are too much data in input to Evidently is suggested to avoid plotting inside a notebook and save results directly to HTML or JSON.


## Creating custom reports

Evidently reports can be customized using custom widgets or entire tabs. Custom widgets building requires deriving a class from the [base Widget class](https://github.com/evidentlyai/evidently/blob/main/src/evidently/dashboard/widgets/widget.py), or from the [base Tab class](https://github.com/evidentlyai/evidently/blob/main/src/evidently/dashboard/tabs/base_tab.py).

## AudioNet integration
In this section we outline how Evidently integration with the AudioNet project has been performed.

### Performance parameters extraction

First of all, we need to get original and predicted validation set labels, so we edit the AudioNet class adding a method able to return validation and test results

```python
class AudioNet(pl.LightningModule):
    
    def __init__(self, n_classes = 50, base_filters = 32):
        # [...]

        self.test_results = tuple()
        self.validation_results = tuple()
    
    # [...]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = torchmetrics.functional.accuracy(y_hat, y)
        self.log('acc', acc, on_epoch=True, prog_bar=True)

        self.validation_results = acc, y_hat, y

        return acc
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = torchmetrics.functional.accuracy(y_hat, y)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)

        self.results = acc, y_hat, y
```

Next, we modified the `train()` function to return these values

```python
def train():
    # [...]

    # Initialize the network
    audionet = AudioNet()
    trainer = pl.Trainer(gpus=0, max_epochs=2)
    trainer.fit(audionet, train_loader, val_loader)

    result = trainer.test(audionet, test_loader)

    (val_accuratezza, val_y_hat, val_y, val_x) = audionet.get_validation_results()
    (accuratezza, y_hat, y, x) = audionet.get_results()

    return accuratezza, y_hat, y, val_accuratezza, val_y_hat, val_y
```

### Dashboard preparation and plotting
Now we can take values returned by the train() function and use them with Evidently. As explained before, Dashboard can be prepared directly passing the DataFrames reference and production.

```python
if __name__ == "__main__":
    acc, y_hat, y, val_acc, val_y_hat, val_y, = train()

    # [...]

    reference = {"target": val_y.tolist(), "prediction": val_y_hat.tolist()}
    production = {"target": y.tolist(), "prediction": y_hat.tolist()}

    df_reference = pd.DataFrame(dict_reference)
    df_production = pd.DataFrame(dict_production)

    audionet_report = Dashboard(tabs=[RegressionPerformanceTab()])
    audionet_report.calculate(df_reference, df_production, column_mapping = None)
    audionet_report.show()
```


### Google Colab integration sidenote

The work described in this repository has been performed on Google Colab due to the lack of a suitable GPU.
Right now it seems that PyTorch-Lightning and TorchAudio version bundled in the Colab's runtime are not fully compatible. So, in order to setup correctly PyTorch-Lightning and TorchAudio in this platform we used directly the git version of PyTorch-Lightning, which seems not affected by the issue described.

```sh
!pip install git+https://github.com/PyTorchLightning/pytorch-lightning
```

For reference, the branch used for this work is the `290fb466de1fcc2ac6025f74b56906592911e856`, along with `torchaudio 0.10.0+cu111`.

## Conclusions and further observations

In this section we sum up a few more observations and tips that could be useful:

* in case of datasets particularly big is useful to perform dashboard and report plotting directly to HTML or JSON files;
* Evidently requires data passed as DataFrame using lists, so PyTorch's tensor should be converted to lists (with the tolist() method);
* the ClassificationPerformance tab require that all labels of the classification problem appear in the reference DataFrame.

In conclusion, Evidently is a powerful open source tool to plot performance metrics and compare machine learning models. It can also be customized by providing JSON description files.

Enjoy.
