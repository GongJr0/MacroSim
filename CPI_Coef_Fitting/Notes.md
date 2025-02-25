# Regression Based CPI Predictions
## Introduction to CPI in Economic Simulation
### CPI on a Nutshell 
__Consumer Price Index__ (and it's variations) is a parameter that tracks the price levels of 
selected goods. (aka. basket) CPI records the price inflation/deflation in the subset of goods,
often selected to reflect the expenses of the ordinary (median) consumer to reflect the magnitude 
of nominal inflation felt by the public. Hence, it is a popular measure that is discussed in media
and economic news sources.

### Purpose of CPI in Macroeconomic Modelling
CPI is closely tied to __Money Demand__ (MD) in a truly simple way; if one needs to spend more they need
more in their wallet. We can visualize the effects in an exaggerated scenario to form the idea of this relation.
Imagine the price of goods (CPI) is stable while a huge pandemic happens, leading to massive lay-offs. I believe this
shouldn't be much of a foreign concept. However, we will make two large assumption that will diversify our fictional
scenario from the reality. In our world, governments will not do anything to stabilize the situation. Furthermore,
the evil corporations in our world will be hellbent on one thing, no price cuts even if it means people will starve to death.

In this world, what would happen to all the people that got laid off but are in dire need of groceries after a few months 
of eating through their savings? They would use their almighty credit cards. In other words, they would __demand money__ 
from their banks to (hopefully) pay it back in the future. 

You might have noticed we forced the CPI to be stable throughout the events; in reality, it should've decreased to accommodate the decrease in income. People won't borrow 
indefinitely, which forces the sellers to meet at a middle (equilibrium) price where the buyers are willing to buy 
and the sellers are (reluctantly) willing to sell.

Let's focus on the part where I mentioned people will demand money; on such a large scale shock, some amount of money
would have been introduced in circulation. And if there's one thing we all know, adding more money to an economy  creates
__inflation__. Although the effect is orders of magnitudes milder then what was described, this is the core principle 
that makes CPI a pillar of real-world MD modelling.

### What are the alternatives to CPI
The main alternative to CPI in a demand modelling scenario is the __Producer Price Index__. (PPI) The difference between
these two can be boiled down to what's in the basked of goods. CPI focuses on core needs of the public and defines their
purchasing power when paired with the wages. PPI baskets, on the other hand, reflect the cost of producing said goods and
services. The main reasoning in choosing CPI over PPI is the prioritization of the public when structuring macroeconomic
policies. Federals banks most prominently use CPI when setting and analysing their inflation targets, therefore modelling
MD through CPI reflects a more accurate picture of how the reaction may be to changes in price levels.

## Predicting CPI
This engine's approach to CPI is one that follows linear coefficients. By selecting some parameters that go into the 
engine, we can define a linear equation of said parameters to define a constructor of CPI. In a general form, 
$CPI = \alpha X_1 + \beta X_2 + ...+\gamma X_n + \epsilon$ where $\{\alpha, \beta, \gamma\}$ are coefficients that define how big
of a role each parameter $\{X_1, X_2, ..., X_n\}$ play in determining CPI. Of course, these coefficients will differ for
each country, creating the need to remodel the equation for each country (or fictional situation) we want to simulate.
However, this gives us a __deterministic and consistent__ way of defining a key variable. The last term of this equation,
$\epsilon$ serves the purpose of adding randomness and is not a part of the regression process, as a note, it comes from 
a random sample that follows a dynamically centered normal distribution with an adjustable scale (variance).

By forming a linear definition, (and potentially sacrificing some accuracy) we reach a set of coefficients that are 
directly usable in the simulation loop without further computations or data transformations. Immediately generating 
linear coefficients with linear regression does not produce desirable accuracy scores, likely due to the lack of data 
when looking into a relatively short time window for training, and the constraints we have in regard to feature scaling, 
normalization, and transformation. Having nothing more than raw data (and not enough of it) is a prime example of what 
not to do in linear regression scenarios, the constraints of this approach practically forces this approach to preserve
simplicity and computational efficiency. 

To account for said limitations, a model distillation process is used with a robust, non-linear model. (namely Random 
Forest) The steps to attain linear coefficients with said approach are:

1. Train a  RandomForest (RF) instance on the raw data
2. Confirm sufficient model performance
3. Train a linear model, GAM, or Symbolic regressor on the RandomForest output
4. Fetch coefficients after confirming linear models performance is similar to RF

It's already confirmed that within the defined restraints, RF is by far the best performing model with
a 97.2% prediction accuracy, and it's inherent outlier smoothing. The distillation process, however, is still in
development. Linear models struggle to capture the complex relationships of RF, leading to inconsistent results at best
and unusable ones at worst. Using a symbolic regressor to create a (potentially) non-linear expression that defines the
RF prediction space will most probably be the best approach to preserve accuracy.