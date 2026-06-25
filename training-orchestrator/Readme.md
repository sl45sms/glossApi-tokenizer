bash that run in the loop to start QWEN as agent to monitor the training.

look at `nohup` and `&` to run the command in the background.
may also need to use `bg` and `disown` to manage the background process.


```bash


1. start QWEN as agent
2. start the training process
3. angent monitor the training process and report the status, decide whether to continue or stop the training based on the status.
4. continue the loop until the training is completed or stopped by the agent.

