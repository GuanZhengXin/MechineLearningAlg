from wxpy import *

bot = Bot(cache_path=True)
mon = bot.friends().search('老妈')[0]
pjj = bot.friends().search('pjj', sex=FEMALE)[0]
hsq = bot.friends().search('洪仕强', sex=MALE)[0]
mon.send('妈,你好')
pjj.send('hello,my love girl. met you is my fortunate')
hsq.send('小强,你好')

@bot.register(mon)
def reply_my_friend(msg):
    return '此人已丢失,请去宁波寻找'

@bot.register(pjj)
def reply_my_friend(msg):
    return '此人已丢失,请去宁波寻找'

@bot.register(hsq)
def reply_my_friend(msg):
    return '此人已丢失,请去宁波寻找'

embed()