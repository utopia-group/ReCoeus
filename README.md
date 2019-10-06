# Coeus Learner

This repository contains the reinforcement learning source code for our paper Relational Verification using Reinforcement Learning published at OOPSLA'19. Source code for the relational verifier part is released as [a separated repository](https://github.com/utopia-group/Coeus).

## Building from source

The RL engine is written in [Python](https://www.python.org), leveraging the [pytorch](https://pytorch.org) library for learning and the [tensorboardx](https://github.com/lanpa/tensorboardX) library for logging and visualization. Everything is dynamic so there's nothing to build here :)

However, to correctly set up all the dependencies a bit of work is required. The recommended way of dependency management in Python is to set up a [virtual environment](https://docs.python.org/3/library/venv.html):

```bash
> git clone https://github.com/utopia-group/ReCoeus.git
> mkdir venv
> python3 -m venv venv
> source venv/bin/activate
```

Now, we can install the libraries we need:

```bash
> pip install torch tensorboardx
```

## Training

First, build a development version of `coeus server` ([instructions](https://github.com/utopia-group/Coeus/blob/master/README.md#coeus-server)). Pick a set of training benchmarks, a set of testing benchmarks and a port number. Run the following command in the verifier's project root (we use port 12345 in this example):

```bash
> ./server.sh training_benchmarks/ testing_benchmarks/ -v -a localhost -p 12345
```

NOTE: As a matter of fact, the testing benchmarks are not relevant here: our training script will never look at them. An early prototype of the training script used to measure training progress using success rate of the test set. We no longer do that now but the legacy interface remains.

You can also use `coeus server` but you'll want to specify the resource limits on your own. By default, conflict analysis on the training server side is disabled. Passing additional `--mc XXX` flag to the server to enable it, where `XXX` denotes the size of the conflict cache (10K is often more than necessary). 

Next, pick a directory where the trained model will be stored, and run the following command from the virtual environment we just set up:

```bash
> python training.py -a localhost -p 12345 -n 10000 -o model_output/
```
Here we use `-n` to specify after how many episodes are we going to stop. If you don't know a good number to set, just use a very large number, and later terminate the training manually. To monitor training progress, fire up another terminal, run `tensorboard model_output`, and open the link you get there in your browser.

## Testing

After the training terminates, the trained model will be available at `model_output/`. To use the model to guide proof search, we'll again rely on client-server socket communication except that this time the Python side will be the server and the OCaml side will be the client.

Use the following command to start a search server from the virtual environment:

```bash
> python search_server.py model_output/ -a localhost -p 12346
```
We use `-p` to set the port number to 12346. Next, connect to this server from the OCaml side using `coeus run`:

```bash
> coeus run XXX.coeus --proof guided -a localhost -p 12346 -v
```
There are 3 search strategies in `coeus run` that relies on socket connection to an existing search server: `guided` (single-rollout), `repeatguided` (multi-rollout), and `guidedexhaustive` (guided exhaustive search, which is *the* algorithm proposed in the paper). 