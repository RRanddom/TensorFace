import vggface
from pprint import pprint
import tensorflow as tf
input_placeholder = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
network = vggface.VGGFace()



ses = tf.InteractiveSession()
network.load(ses,input_placeholder)

output = network.eval(feed_dict={input_placeholder:vggface.load_image('test/ak.png')})[0]
pprint(sorted([(v,network.names[k]) for k,v in enumerate(output)],reverse=True)[:10])


output = network.eval(feed_dict={input_placeholder:vggface.load_image('test/IMG_0647.jpg')})[0]
pprint(sorted([(v,network.names[k]) for k,v in enumerate(output)],reverse=True)[:10])