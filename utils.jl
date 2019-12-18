## we implement a method to load the original MAT file of then
## binarized MNIST

using MAT
using MLDataUtils
using Distributions
import Distributions: logpdf
using MLDatasets: MNIST


img(x) = Gray.(permutedims(reshape(x, 28, 28)))


function load_binarized_mnist(batch_size, flatten)
    # we start by loading the training set
    file = open("Original_MNIST_binarized/binarized_mnist_train.amat", "r")
    train_imgs = readlines(file)
    #C =  findall(x->x=="1", vec(train_imgs[2,:]))
    #@show C
    train_imgs = [parse.(Float64, split(train_imgs[i]," ")) for i in 1:size(train_imgs, 1)]
    example = train_imgs[1]
    save("check_while_load_fin.png", img(example))
    #C =  findall(x->x==1.0, vec(train_imgs[2,:][1]))
    #@show C
    #train_imgs = permutedims(reshape(hcat(train_imgs...), (length(train_imgs[1]), length(train_imgs))))
    train_imgs = hcat(train_imgs...)
    @show(size(train_imgs))
    example = train_imgs[:,1]
    save("check_while_load_fin2.png", img(example))
    #C =  findall(x->x==1.0, vec(train_imgs[2,:]))
    #@show C
    @show(size(train_imgs))

    # valid test
    file = open("Original_MNIST_binarized/binarized_mnist_valid.amat", "r")
    valid_imgs = readlines(file)
    valid_imgs = [parse.(Float64, split(valid_imgs[i]," ")) for i in 1:size(valid_imgs, 1)]
    valid_imgs = hcat(valid_imgs...)
    #valid_imgs = permutedims(reshape(hcat(valid_imgs...), (length(valid_imgs[1]), length(valid_imgs))))
    #valid_imgs = permutedims(reshape(hcat(valid_imgs...), (length(valid_imgs), length(valid_imgs[1]))))
    #@show(size(valid_imgs))

    # test set
    file = open("Original_MNIST_binarized/binarized_mnist_test.amat", "r")
    test_imgs = readlines(file)
    test_imgs = [parse.(Float64, split(test_imgs[i]," ")) for i in 1:size(test_imgs, 1)]
    test_imgs = hcat(test_imgs...)
    #test_imgs = permutedims(reshape(hcat(test_imgs...), (length(test_imgs[1]), length(test_imgs))))
    #test_imgs = permutedims(reshape(hcat(test_imgs...), (length(test_imgs), length(test_imgs[1]))))

    #@show(size(test_imgs))

    # flatten
    if flatten
        # we want to reshape them into (L, BS)
        # training set
        x_train = train_imgs
        # validation set
        x_valid = valid_imgs
        # test set
        x_test = test_imgs

    else
        # we want to return the images (H,W,C,n) where n = |set|
        # training set
        x_train = reshape(train_imgs, 28, 28, 1, size(train_imgs, 2))
        # validation set
        x_valid = reshape(valid_imgs, 28, 28, 1, size(valid_imgs, 2))
        # test set
        x_test = reshape(test_imgs, 28, 28, 1, size(test_imgs, 2))

    end

    return x_train, x_valid, x_test
end


#const batch_size = 64
#x_train, x_valid, x_test = load_binarized_mnist(batch_size, false)

## function to load MNIST (credit: Jesse Bettencourt)
function loadMNIST(batch_size)
    # we use MLDataUtils LabelEnc for the one-hot-conversion
    onehot(labels_raw) =  convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
    # load MNIST from the library
    imgs, labels_raw = MNIST.traindata();
    # process the images into (H,W,C,BS) batches
    @show(size(imgs))
    x_train = reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3))
    @show(size(x_train))
    x_train = batchview(x_train, batch_size)
    @show(size(x_train))
    # one-hot and batch the labels
    y_train = onehot(labels_raw)
    y_train = batchview(y_train, batch_size)

    ## we should do the same with the test set
    imgs, labels_raw = MNIST.testdata();
    # process the images into (H,W,C,BS) batches
    x_test = reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3))
    x_test = batchview(x_test, batch_size)
    # one-hot and batch the labels
    y_test = onehot(labels_raw)
    y_test = batchview(y_test, batch_size)

    return x_train, y_train, x_test, y_test
end

#const batch_size = 64
#x_train, x_valid, x_test = load_binarized_mnist(batch_size, true)
#@show size(x_train)
#x_train, y_train, x_test, y_test = loadMNIST(batch_size)
