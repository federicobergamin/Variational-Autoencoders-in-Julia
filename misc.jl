using Flux, Statistics
using Flux: throttle, params, binarycrossentropy, gradient, @epochs
#flux function for update parameters
using Flux.Optimise: update!
using Juno: @progress
using MLDataUtils
using Distributions
import Distributions: logpdf
using Images

# include("utils.jl")
# batch_size = 64
# x_train, x_valid, x_test = load_binarized_mnist(batch_size, true)
# @show size(x_train[:, 2])
# #x_train = x_train .> 0.5
#
# N = size(x_train, 2)
# x_train_batch = [x_train[:,i] for i in Iterators.partition(1:N,batch_size)]
#
# img(x) = Gray.(reshape(x, 28, 28))
# example = x_train[:,1]
# save("/Users/federicobergamin/Desktop/Research_Assistant_DTU_Compute/Code/Variational_Autoencoders/Julia VAE/prova.png", img(example))
# #
#
# c = vec([1 2 3 4 5 6 7 7 8])
#
# for i in c
#     print(i)
# end
# img = rand(4, 4)
#
# using Plots
#
# plot(img)

img(x) = Gray.(permutedims(reshape(x, 28, 28)))
file = open("Original_MNIST_binarized/binarized_mnist_train.amat", "r")
train_imgs = readlines(file)
#C =  findall(x->x=="1", vec(train_imgs[2,:]))
#@show C
train_imgs = [parse.(Float32, split(train_imgs[i]," ")) for i in 1:size(train_imgs, 1)]

@show train_imgs[1,:]

example = train_imgs[3]
save("check3.png", img(example))
#C =  findall(x->x==1.0, vec(train_imgs[2,:][1]))
#@show C
#train_imgs = permutedims(reshape(hcat(train_imgs...), (length(train_imgs[1]), length(train_imgs))))
train_imgs = permutedims(reshape(hcat(train_imgs...), (length(train_imgs), length(train_imgs[1]))))
#C =  findall(x->x==1.0, vec(train_imgs[2,:]))
#@show C
@show(size(train_imgs))


a = [1 2 3 4 5 6 7 8 9]

size(a)
b = permutedims(reshape(a, 3, 3))



img(x) = Gray.(permutedims(reshape(x, 28, 28)))
x_train, x_valid, x_test = load_binarized_mnist(batch_size, true)
example = x_train[:,1]
save("original2.png", img(example))




##### solve this stupid PROBLEMS
file = open("Original_MNIST_binarized/binarized_mnist_train.amat", "r")
train_imgs = readlines(file)
#C =  findall(x->x=="1", vec(train_imgs[2,:]))
#@show C
train_imgs = [parse.(Float32, split(train_imgs[i]," ")) for i in 1:size(train_imgs, 1)]
size(train_imgs)
example = train_imgs[1]
save("check_while_load.png", img(example))

# i have train images which is a vector of 50000 vecotrs of size 784
# i want to transform it in a matrix
hcat(train_imgs...)
example = train_imgs[1]
save("check_while_load23.png", img(example))





#C =  findall(x->x==1.0, vec(train_imgs[2,:][1]))
#@show C
#train_imgs = permutedims(reshape(hcat(train_imgs...), (length(train_imgs[1]), length(train_imgs))))
train_imgs = permutedims(reshape(train_imgs, (length(train_imgs), length(train_imgs[1]))))
@show(size(train_imgs))
example = train_imgs[:,1]
save("check_while_load2.png", img(example))



rand(Float32)

rand(Float32, 10)


eps(Float32)

a = [1 2 3 4 5 5 6 -1]
if !minimum(a.<0)
    print("pam")
end


mu_ = vec([1 2 3 4])
log_ = vec([2 2 2 2])
x = vec([5 5 5 5])


function log_gaussian(x, mu, log_var)
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x. (Univariate distribution)
    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    #log_pdf = - 0.5 * log(2 * pi) .- log_var ./ 2 - (x .- mu)^2 ./ (2 * exp.(log_var))
    log_pdf =  @. - 0.5 * log(2 * pi) - log_var / 2 - (x - mu)^2 / (2 * exp(log_var))
    # print('Size log_pdf:', log_pdf.shape)
    return log_pdf
end

log_gaussian(x, mu_, log_)

log_ - mu_

function log_standard_gaussian(x)
    #return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)
    A =  -0.5f0 * log(2 * pi) .- x.^2 / 2
    #@show(A)
    # alternatively, we can do
    #Norm = Normal(0)
    #A = log(pdf(Norm, x))
    #@show(A)
    return A
end

function _kl_divergence(z, mu_q, log_var_q, mu_p = nothing, log_var_p = nothing)
    # instead of computing it analytically, we can estimate it
    # using monte carlo methods
    qz = log_gaussian(z, mu_q, log_var_q)
    @show(qz)
    ## we should do the same with p
    if isnothing(mu_p) & isnothing(log_var_p)
        pz = log_standard_gaussian(z)
        @show(pz)
    else
        pz = log_gaussian(z, mu_p, log_var_p)
    end
    kl = qz - pz
    return kl
end

_kl_divergence(x, mu_, log_)

log_standard_gaussian(0)
