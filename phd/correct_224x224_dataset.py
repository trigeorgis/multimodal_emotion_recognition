import tensorflow as tf
import numpy as np
import pdb

def read_labeled_image_list(image_list_file):
	"""Reads a .txt file containing pathes and labeles
        Args:
        image_list_file: a .txt file with one /path/to/image per line
        label: optionally, if set label will be pasted after each line
        Returns:
        List with all filenames in file image_list_file
	"""
	f = open(image_list_file, 'r')
	filenames = []
	labelsv_me = []
	labelsv_thanos = []
	labelsv_hen = []
	labelsv_panos = []

	labelsa_me = []
	labelsa_thanos = []
	labelsa_hen = []
	labelsa_panos = []

	for line in f:
		line = line.rstrip()
		inputs = line[:].split(' ')
		filenames.append(inputs[0])
		labelsv_me.append(float(inputs[1]))
		labelsv_thanos.append(float(inputs[3]))
		labelsv_hen.append(float(inputs[5]))
		labelsv_panos.append(float(inputs[7]))
		labelsa_me.append(float(inputs[2]))
		labelsa_thanos.append(float(inputs[4]))
		labelsa_hen.append(float(inputs[6]))
		labelsa_panos.append(float(inputs[8]))

	labels = [list(a) for a in zip(labelsv_me, labelsa_me,labelsv_thanos, labelsa_thanos,labelsv_hen, labelsa_hen,labelsv_panos, labelsa_panos)]
	return filenames, labels

def read_labeled_land_list(image_list_file):
	"""Reads a .txt file containing pathes and labeles
        Args:
        image_list_file: a .txt file with one /path/to/image per line
        label: optionally, if set label will be pasted after each line
        Returns:
        List with all filenames in file image_list_file
	"""
	f = open(image_list_file, 'r')
	filenames = []
	

	for line in f:
		line = line.rstrip()
		inputs = line[:].split(' ')
		filenames.append(inputs[0])
	
	return filenames


def many_read_labeled_image_list(file1,file2,file3,file4,file5):
	"""Reads a .txt file containing pathes and labeles
        Args:
        image_list_file: a .txt file with one /path/to/image per line
        label: optionally, if set label will be pasted after each line
        Returns:
        List with all filenames in file image_list_file
	"""
	fi = []
	fi.append(file1)
	fi.append(file2)
	fi.append(file3)
	fi.append(file4)
	fi.append(file5)
	filenames = []
	labelsv = []
	labelsa = []
	
	for fil in fi:
		f = open(fil, 'r')
		for line in f:
			line = line.rstrip()
			filename, labelv , labela = line[:].split(' ')
			filenames.append(filename)
			labelsv.append(float(labelv))
			labelsa.append(float(labela))
	labels = [list(a) for a in zip(labelsv, labelsa)]
	return filenames, labels


def read_images_from_disk(input_queue,images):
    """Consumes a single filename and label as a ' '-delimited string.
        Args:
        filename_and_label_tensor: A scalar string tensor.
        Returns:
        Two tensors: the decoded image, and the string label.
        """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    
    
    example = tf.image.resize_images(example, tf.convert_to_tensor([150,150]))
    example = tf.image.resize_image_with_crop_or_pad(example, 224,224)
 
    return example,label

def read_images_lands_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
        Args:
        filename_and_label_tensor: A scalar string tensor.
        Returns:
        Two tensors: the decoded image, and the string label.
        """
    label = input_queue[1]

    file_contents = tf.read_file(input_queue[0])
 
    filename_queue = tf.train.string_input_producer([input_queue[2]])

    reader = tf.TextLineReader()
    _, line = reader.read(filename_queue)

    line_batch = tf.train.batch([line], batch_size=68)
    col1,col2 = tf.decode_csv(line_batch,record_defaults=[tf.constant([],dtype=tf.float32),tf.constant([],dtype=tf.float32)])
                                            
    features = tf.pack([col1,col2],1)

    example = tf.image.decode_jpeg(file_contents, channels=3)
    
    
    example = tf.image.resize_images(example, tf.convert_to_tensor([150,150]))
    example = tf.image.resize_image_with_crop_or_pad(example, 224,224)
 
    return example,label,features


def read_batch_images_from_disk(input_queue,whole_size):
	"""Consumes a single filename and label as a ' '-delimited string.
        Args:
        filename_and_label_tensor: A scalar string tensor.
        Returns:
        Two tensors: the decoded image, and the string label.
	"""
	examples = []
	labels = []
	for iter in range(whole_size):
		label = tf.gather(input_queue[1], [iter])
		labels.append(label)
		im = tf.gather(input_queue[0], [iter])
		im = tf.reshape(im,[])
		file_contents = tf.read_file(im)
		example = tf.image.decode_jpeg(file_contents, channels=3)
		example = tf.image.resize_images(example, tf.convert_to_tensor([150,150]))
		example = tf.image.resize_image_with_crop_or_pad(example, 224,224)
		examples.append(example)
	return examples,labels

def read_batch_images_from_disk2(input_queue,same,different):
	"""Consumes a single filename and label as a ' '-delimited string.
        Args:
        filename_and_label_tensor: A scalar string tensor.
        Returns:
        Two tensors: the decoded image, and the string label.
	"""
	examples = []
	labels = input_queue[1]
	ex = []
	in_im = tf.reshape(input_queue[0],[-1])
	#in_lab = tf.reshape(input_queue[1],[-1,2])
	for i in range(different*same):
		file_contents = tf.read_file(in_im[i])
		example = tf.image.decode_jpeg(file_contents, channels=3)
		example = tf.image.resize_images(example, tf.convert_to_tensor([150,150]))
		example = tf.image.resize_image_with_crop_or_pad(example, 224,224)
		examples.append(example)
	examples = tf.reshape(examples,[different,same,224,224,3])		
	return examples,labels

