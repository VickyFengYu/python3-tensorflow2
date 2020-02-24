
Python Tensorflow Recommendation System
===================


## <i class="icon-file"></i> Install Tensorflow

```
$ brew install python3

$ brew link --overwrite python
Linking /usr/local/Cellar/python/3.7.6_1... 28 symlinks created

$ sudo easy_install pip
$ pip install tensorflow

```

Test install

```
$ python
Python 3.7.6 (default, Dec 30 2019, 19:38:26) 
[Clang 11.0.0 (clang-1100.0.33.16)] on darwin
Type "help", "copyright", "credits" or "license" for more information.

>>> import tensorflow as tf
>>>  tf.add(1, 2).numpy()

>>> hello = tf.constant('Hello, TensorFlow!')
>>> hello.numpy()
b'Hello, TensorFlow!'

```

## <i class="icon-file"></i> Import Data Set

[Download ml-latest-small.zip](https://grouplens.org/datasets/movielens/)
