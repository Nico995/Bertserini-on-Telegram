from bertserini_on_telegram.utils.translate import AutoTranslator

translator = AutoTranslator()

print(translator.translate('Ciao, come ti chiami?', None, 'en_XX'))