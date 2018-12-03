import matplotlib.pyplot as plt
import numpy as np

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
            print "Expected utility (without normalizing) less than 0 for user of type ",user_type, " and pdfM =",pdfM
    return max(0.,exp_util)

if (__name__ == "__main__"):
    arg = "all-0-1-uniform-3d"
    #2d plot
    if (arg == "all-0-1-uniform-2d"):
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
        print to_plot
        fig = plt.figure("Optimal Strategy Bid")
        s = fig.add_subplot(1,1,1,xlabel="User Type",ylabel="P[M = 1st Price]")
        im = s.imshow(to_plot,
                extent = (0,1,0,1),
                origin='lower')
        fig.colorbar(im)
        plt.show()



    elif (arg == "all-0-1-uniform-3d"):
        #Assuming F is the same for all, and U[0,1]
        N = 2
        first_price_const = (0,1.,0,2.)
        second_price_const = (0,0.5,0,1.)
        lottery_const = (0.,1.,(1./N),0.)
        
        Delta_prob = 0.01 #resolution of probabilities
        NUM_TYPES = 10 #number of types, chosen uniformly in between 0 and 1 
        const_list = (first_price_const,second_price_const,lottery_const)

        fig = plt.figure("Optimal Strategy Bid")
        im = None 
        for i,user_type in enumerate(np.arange((1./NUM_TYPES),1.+(1./NUM_TYPES),(1./NUM_TYPES))):
            #Calculate values
            to_plot = []
            for prob_first_price in np.arange(0,1 +Delta_prob,Delta_prob):
                to_plot.append([])
                for prob_second_price in np.arange(0,1 + Delta_prob, Delta_prob):

                    prob_lottery = 1. - prob_first_price - prob_second_price
                    if (prob_lottery < 0 or prob_lottery >= 1):
                        val_list = to_plot[-1]
                        val_list.append(np.nan)
                    else:
                        probM = (prob_first_price, prob_second_price,prob_lottery)
                        val = calc_exp_utility(const_list,probM,user_type)
                        print "Prob First Price (%.2f),Prob Second Price (%.2f), User_type (%.2f)-> Exp Utility (%.2f) "%(prob_first_price,prob_second_price,user_type,val)
                        val_list = to_plot[-1]
                        val_list.append(val)

            #Construct a plot
            ax = fig.add_subplot(2,5,(i+1),xlabel="P[M = 2nd Price]",ylabel="P[M = 1st Price]")
            ax.set_title("User Type %.2f"%user_type)
            #print to_plot
            current_cmap = plt.cm.get_cmap()
            current_cmap.set_bad(color='white')
            im = ax.imshow(to_plot,
                    extent = (0,1,0,1),
                    origin='lower')
                    #vmin=0,
                    #vmax=1)
       
        #Show plots

            fig.colorbar(im)
        plt.show()

