import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
DEBUG = True
"""
    calc_exp_utility: Calculates the maximum expected utility of an agent in a bid-affine environment
        inputs
        const_list = list of 4-tuples containing constants for each mechanism
        pdfM = list of probabilities of each mechanism
        user_type = non-negative real value of user's type
"""
def calc_exp_utility(const_list,pdfM,user_type):
    #sanity check
    assert(len(const_list) == len(pdfM))
    assert(len(const_list) > 0)
    #assert we get c0,c1,c2,c3
    for const in const_list:
        assert(len(const) == 4)
    #assert positive user type
    assert(user_type >= 0.)

    #Calc max expected utility
    num = 0.
    denom = 0.

    for (const,pdfm) in zip(const_list,pdfM):

        #pdfm is prob of mechanism m
        #const
        (c0,c1,c2,c3) = const
        num = num + (c3*(user_type - c0) - c1*c2)*pdfm
        denom = denom + 2*c1*c3*pdfm

    assert(denom > 0.)
    exp_util = num/denom
    
    if (DEBUG):
        if (exp_util < 0.):
            print "Expected utility (without normalizing) less than 0 for user of type ",user_type
    return max(0.,exp_util)

if (__name__ == "__main__"):
    F = "all-0-1-uniform"
    if (F == "all-0-1-uniform"):
        #Assuming F is the same for all, and U[0,1]
        first_price_const = (0,1.,0,2.)
        second_price_const = (0,0.5,0,1.)
        
        Delta = 0.05 #resolution of plot
        const_list = (first_price_const,second_price_const)
        to_plot = []
        for prob_first_price in np.arange(0,1 +Delta,Delta):
            to_plot.append([])
            for user_type in np.arange(0,1+Delta,Delta):
                probM = (prob_first_price,1.0 - prob_first_price)
                val = calc_exp_utility(const_list,probM,user_type)
                print "Prob First Price (%.2f),User_type (%.2f)-> Exp Utility (%.2f) "%(prob_first_price,user_type,val)
                val_list = to_plot[-1]
                val_list.append(val)


        #ax = sns.heatmap(np.array(to_plot))
        #plt.show()
        fig = plt.figure()
        s = fig.add_subplot(1,1,1,xlabel="User Type",ylabel="P[M = 1st Price]/1 - P[M = 2nd Price]")
        im = s.imshow(to_plot,
                extent = (0,1,0,1),
                origin='lower')
        fig.colorbar(im)
        plt.show()

