import matplotlib.pyplot as plt

from libs import *

def plot_wordcloud(msg_cloud):
    plt.figure(figsize=(16, 10))
    plt.imshow(msg_cloud, interpolation='bilinear')
    plt.axis('off')  # turn off axis
    plt.show()


url = 'https://raw.githubusercontent.com/ShresthaSudip/SMS_Spam_Detection_DNN_LSTM_BiLSTM/master/SMSSpamCollection'
messages = pd.read_csv(url, sep='\t', names=['labels', 'message'])

# Visualize data
# print(messages[:3])

# exploring data
# print(messages.describe())
duplicatedRow = messages[messages.duplicated()]
# print(duplicatedRow[:5])
# print(messages.groupby('labels').describe().T)

# exploring data by labels groups by creating a WordCloud and a barchart
# Get all the ham and spam emails
ham_msg = messages[messages.labels == 'ham']
spam_msg = messages[messages.labels == 'spam']

# Create numpy list to visualize using WordCloud
ham_msg_text = " ".join(ham_msg.message.to_numpy().tolist())
spam_msg_text = " ".join(spam_msg.message.to_numpy().tolist())

# Wordcloud of ham messages -> to check most popular word in text
WordCloud_init = WordCloud(width=520, height=260, stopwords=STOPWORDS, max_font_size=50, background_color='black', colormap='Blues')

ham_msg_cloud = WordCloud_init.generate(ham_msg_text)
plot_wordcloud(ham_msg_cloud)

spam_msg_cloud = WordCloud_init.generate(spam_msg_text)
plot_wordcloud(spam_msg_cloud)

# we can observe imbalance data here
plt.figure()