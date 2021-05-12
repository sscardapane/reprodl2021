# Reproducible Deep Learning
## Exercise 3: Data versioning with DVC
[[Official website](https://www.sscardapane.it/teaching/reproducibledl/)] [[Slides](https://docs.google.com/presentation/d/1jUFz212lZvwqDibiCRoOcm-40ANPXI1dKlF8t7PD1Is/edit?usp=sharing)] [[DVC Website](http://dvc.org/)]

## Objectives for the exercise

- [ ] Adding data versioning with DVC.
- [ ] Using multiple remotes for your DVC repository.
- [ ] Downloading files from a DVC repository.

See the completed exercise:

```bash
git checkout exercise3_dvc_completed
```

## Prerequisites

1. If you have not done so already, uncompress the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) inside the *data* folder.
2. If this is your first exercise, run *train.py* to check that everything is working correctly.
3. Install [DVC](http://dvc.org/):

```bash
pip install dvc
```

4. If you plan on using S3 storage for exercise 3.2, install `boto3`:

```bash
pip install boto3
```

## Exercise 3.1 - Setup a DVC repository

For the first exercise, we setup versioning of the ESC-50 dataset, using a separate folder from the computer. Before starting, read the slides and the [DVC User Guide](https://dvc.org/doc/start/data-and-model-versioning).

1. Initialize the DVC repository. This creates a `.dvc` folder with the necessary configuration.

```bash
dvc init
git add .
```

2. Add the dataset to be versioned from DVC.

```bash
dvc add data/ESC-50
git add data/ESC-50.dvc
```

3. Try removing the dataset and fetching it from the local cache:

```bash
rm -r data/ESC-50
dvc status
dvc checkout
```

4. Add a remote storage for your files. For this part, we simply use a separate folder from the computer. See [the DVC guide](https://dvc.org/doc/command-reference/remote/add) for a full list of possible remote repositories.

```bash
dvc remote add -d localremote <path>
git add .dvc/config
dvc push
```

5. Commit everything, then try to recover the dataset from a fresh clone of the repository:

```bash
git commit -m "Added DVC support"
git push
cd <some-other-folder>
git clone <path-to-repository>
dvc pull
```

## Exercise 3.2 - Setup a more realistic remote

Having a "local remote" DVC storage is, of course, limited. We now investigate setting up a more realistic remote. You can use a number of [possible alternatives](https://dvc.org/doc/command-reference/remote/add). Here, we simulate a simple S3-like storage using [MinIO](https://docs.min.io/docs/minio-quickstart-guide.html).

> :speech_balloon: If you have access to the DGX machine (or any other remote machine), try launching the server remotely! Alternatively, use an SSH or Google Drive storage.

1. Download and launch the MinIO server (modify according to your [operating system](https://docs.min.io/docs/minio-quickstart-guide.html)):

```bash
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
./minio server /data
```

2. From the MinIO dashboard, create a bucket called `esc50`. By default, the access key will be **minioadmin**, but you can change it when starting the server.

3. Setup remote storage on the MinIO server:

```bash
dvc remote add -d minio s3://esc50/
dvc remote modify minio endpointurl <minio-url>
```

> :speech_balloon: Locally, `<minio-url>` will be *http://localhost:9000*.

4. Push everything to the MinIO server:

```bash
export AWS_ACCESS_KEY=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
dvc push
```

> :speech_balloon: Modify the access keys accordingly. On Windows, you can use `set` instead of `export`.

5. Try again point 5 from Exercise 3.1.

## Exercise 3.3 - Download a folder from a DVC repository

You can use DVC to download files and folders from a DVC repository. For example, from outside this repository, try:

```bash
dvc list <path-to-repo> data
```

Download the data folder using the default remote:

```bash
dvc get <path-to-repo> data/ESC-50
```

You can also list and/or get from a GitHub repository. For example, from this repository:

```bash
dvc list https://github.com/sscardapane/reprodl2021.git --rev exercise3_dvc_completed data
```

Try pushing the exercise to your own repository, and getting a file from there.

Congratulations! You have concluded another move to a reproducible deep learning world. :nerd_face:

Move to the next exercise:


```bash
git checkout exercise4_docker
