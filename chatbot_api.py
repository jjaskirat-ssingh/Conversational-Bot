import bot
import web

urls = (
    '/tfidf', 'TFIDF',
)

class TFIDF:
    def GET(self):
        userdata = web.input()
        query=userdata.query
        data = bot.bot_engine(query=query)
        return data

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()