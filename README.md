# Bertserini-on-Telegram

Bertserini-on-Telegram is a pytorch Bertserini implementation with a small utility to plug it in a Telegram bot.

## Installation

Install via [pip](https://pypi.org/project/bertserini-on-telegram/)

```bash
pip install bertserini-on-telegram
```

## Usage

The models and dataloaders are written in PyTorchLightning, and are usable on their own, or via PyTorchLightning CLI.

I included a couple of example scripts in the example folder. Each subfolder contains a python script, a yaml config file and a shell script to show how to run the script.

## Running it on Telegram

The telegram example supposes that the used knows how to create a Telegram bot in the first place. Telegram has a lot of useful resources about it, this is a [good starting point](https://core.telegram.org/bots). 

Once the bot is online, we only need to create a file called telegram_token_id.yaml in the  directory.

#### **`./example/telegram_bottelegram_token_id.yaml`**
```yaml
token_id: <token_id>
```

and run the code.