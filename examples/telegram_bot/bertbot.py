import logging
from pytorch_lightning.utilities.cli import LightningCLI
import telegram
import yaml
from bertserini_on_telegram.utils.base import Question
from bertserini_on_telegram.utils.pyserini import Searcher
from bertserini_on_telegram.data import PredictionDataModule

from telegram import ReplyKeyboardRemove, Update
from bertserini_on_telegram.utils.translate import AutoTranslator
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)
logger.info('Welcome, logging has just started!')

# build the context_retreiving object
searcher = Searcher("wikipedia-dpr")

# build the auto translator object
at = AutoTranslator()

# build the predictor object
logger.info('Loading BERT Pretrained')
cli = LightningCLI(run=False, save_config_callback=None, save_config_overwrite=True)
bert = cli.model

ANSWER = range(1)


def start(update: Update, context: CallbackContext) -> int:
    """Starts the conversation and asks the user about their gender."""

    update.message.reply_text(
        "Hi! I'm a telegram chatbot powered by BERT!\n"
        "If you ask me a question I will try to ansewr you based on a couple of matches found on wikipedia\n"
        "To interact with me it's easy, just start any message with the character '!' and I'll try to answer your question!"
    )

    return ANSWER


def cancel(update: Update, context: CallbackContext) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text(
        'Bye! I hope we can talk again some day.', reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


def answer(update: Update, context: CallbackContext) -> int:
    """Cancels and ends the conversation."""
    
    # Bot is typing
    context.bot.send_chat_action(update.message.chat_id, action=telegram.ChatAction.TYPING)
    user = update.message.from_user
    
    # Create question object
    question_text = update.message.text.lstrip('!').lstrip()
    question_text, langs = at.translate(question_text, src_lang=None, trg_lang='en_XX')

    if not question_text:
        update.message.reply_text(f"Sorry, {langs[0]} is not supported by Mbart50 :(")
        return

    logger.info(f'BERT detected the language: {langs[0]}')
    print('BERT translated the question: ', question_text)

    question = Question(question_text, "en")    
    logger.info(f'User {user.first_name} asked a question: {question_text}')

    # Retrieve contexts
    logger.info(f'I am retreiving context from Wikipedia...')
    contexts = searcher.retrieve(question, 10)
    
    # Create datamodule
    dm = PredictionDataModule(question, contexts, cli.model.hparams.model_name)

    # Predict answer
    cli.trainer.predict(bert, dm)
    answer = bert.answer

    if not answer or len(answer) == 0:
        update.message.reply_text(f'Please try changing the phrasing of your question.')
    else:
        answer, _ = at.translate(answer, src_lang='en_XX', trg_lang=langs[0])
        update.message.reply_text(answer)
        logger.info(f'BERT found an answer to the question! "{answer}"')

    return ANSWER


def main() -> None:
    logger.info('Starting BERTBot')
    """Run the bot."""
    # read the telegram token from the config file (very secret wow)
    with open('./telegram_token_id.yaml', 'r') as f:
        token = yaml.safe_load(f)['token']

    # create the Updater and pass it your bot's token.
    updater = Updater(token)

    # get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # add conversation handler with the states
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            ANSWER: [MessageHandler(Filters.regex("^!"), answer)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)

    # Start the Bot
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
