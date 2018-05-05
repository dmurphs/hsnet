module Lib
    ( forwardPropogate,
      weightsWithInitializer,
      randomWeights,
      sigmoid,
      runBatch,
      updateParameters,
      getLayer,
      runEpochs,
      ActivationFunction (..),
      Activation (..),
      Parameters,
      Gradients,
      Layer (..)
    ) where

import Numeric.LinearAlgebra

-- |The Layer type represents a layer in an artificial neural network
data Layer = Layer { layerParameters  :: Parameters
                   , activationFunction :: ActivationFunction
                   }

-- |Wraps activation function and derivative
data ActivationFunction = ActivationFunction { f  :: Double -> Double
                                             , f' :: Double -> Double
                                             }
-- |Wraps values from layer activations
data Activation = Activation { -- | Values after matrix multiplication but before activation function is applied
                               preActivation     :: Matrix Double
                               -- | Values after activation function is applied
                             , postActivation    :: Matrix Double
                             }

type Parameters = (Matrix Double, Vector Double)
type Gradients = Parameters

-- |Computes values for each layer
forwardPropogate :: Matrix Double -> [Layer] -> [Activation]
forwardPropogate inputs = scanl activate firstActivation
  where
    firstActivation = Activation {preActivation=(numInputRows><numInputCols) [0..], postActivation=inputs}
    numInputRows = rows inputs
    numInputCols = cols inputs

activate :: Activation -> Layer -> Activation
activate previousActivation layer = Activation {preActivation=layerPreActivation, postActivation=layerPostActivation}
  where
    layerPostActivation = cmap layerActivationFunction layerPreActivation
    layerPreActivation = (previousPostActivation <> layerWeights) `add` fromRows (replicate (rows previousPostActivation) layerBiases)
    previousPostActivation = postActivation previousActivation
    (layerWeights,layerBiases) = layerParameters layer 
    layerActivationFunction = (f . activationFunction) layer

-- |Get random weights for a given network structure
randomWeights :: [Int] -> Maybe (IO [Matrix Double])
randomWeights []  = Nothing
randomWeights [x] = Nothing
randomWeights xs  = Just $ traverse (uncurry randn) $ steps xs

-- |Get weights all with initializer value for given network structure
weightsWithInitializer :: Double -> [Int] -> [Matrix Double]
weightsWithInitializer initializer layerSizes = map (\(from,to) -> (from><to) $ repeat initializer) $ steps layerSizes

steps :: [a] -> [(a,a)]
steps [] = []
steps [x] = []
steps (x:y:xs) = (x,y) : steps (y:xs)

-- |Sigmoid activation function
sigmoid :: ActivationFunction
sigmoid = ActivationFunction {f=s,f'=s'}
  where
    s x = 1 / (1 + exp (-x))
    s' x = s x * (1 - s x)

-- |Run a batch for training neural network model
runBatch :: Matrix Double -> Matrix Double -> [Layer] -> [Gradients]
runBatch inputs expected initialParameters = runBatch' (outputPreviousActivations:remainingActivations) (reverse initialWeights) (tail activationDerivatives) outputDelta [(outputWeightGradients, outputBiasGradients)]
  where
    outputBiasGradients = getBiasGradient outputDelta
    outputWeightGradients = getWeightGradients outputDelta outputPreviousActivations
    outputDelta = getOutputDelta expected output outputActivationDerivative 
    (output:outputPreviousActivations:remainingActivations) = reverse activations
    outputActivationDerivative = last activationDerivatives
    activations = forwardPropogate inputs initialParameters
    initialWeights = map (fst . layerParameters) initialParameters
    initialBiases = map (snd . layerParameters) initialParameters
    activationDerivatives = map (f' . activationFunction) initialParameters

runBatch' :: [Activation] -> [Matrix Double] -> [Double -> Double] -> Matrix Double -> [Gradients] -> [Gradients]
runBatch' [input] _ _ _ gradients = gradients
runBatch' activations weights activationDerivatives nextDelta gradients = runBatch' (previousActivations:remaining) currentWeights remainingActivationDerivatives currentDelta (gradient:gradients)
  where
    gradient = (getWeightGradients currentDelta previousActivations, getBiasGradient currentDelta)
    currentDelta = getDelta currentActivations nextDelta nextWeights currentActivationDerivative
    (currentActivations:previousActivations:remaining) = activations
    (nextWeights:currentWeights) = weights
    (currentActivationDerivative:remainingActivationDerivatives) = activationDerivatives

getDelta :: Activation -> Matrix Double -> Matrix Double -> (Double -> Double) -> Matrix Double
getDelta activation nextDelta nextWeights activationDerivative = (nextDelta <> tr nextWeights) * cmap activationDerivative (preActivation activation)

getBiasGradient :: Matrix Double -> Vector Double
getBiasGradient deltaMatrix = fromList $ map (\column -> sum column / fromIntegral (length column)) columns
  where
    columns = map toList $ toColumns deltaMatrix

getOutputDelta :: Matrix Double -> Activation -> (Double -> Double) -> Matrix Double
getOutputDelta expected output activationDerivative = (postActivation output - expected) * cmap activationDerivative (preActivation output)

getWeightGradients ::  Matrix Double -> Activation -> Matrix Double
getWeightGradients delta previousActivations = tr previousPostActivation <> delta
  where
    previousPostActivation = postActivation previousActivations

-- updateAdjustments :: [Parameter] -> [Parameter] -> Double -> Double -> [Parameter]
-- updateAdjustments adjustments gradients learningRate momentum = zipWith updateWeightAdjustments adjustments gradients
--   where
--     updateWeightAdjustments :: Parameter -> Parameter -> Parameter
--     updateWeightAdjustments (weightAdjustment,biasAdjustment) (weightGradient,biasGradient) =
--       (
--         cmap (*momentum) weightAdjustment + cmap (*learningRate) weightGradient,
--         cmap (*momentum) biasAdjustment + cmap (*learningRate) biasGradient
--       )

updateParameters :: Double -> Double -> [Gradients] -> [Parameters] -> [Parameters]
updateParameters regularizationTerm learningRate = zipWith updateParameters
  where
    updateParameters :: Gradients -> Parameters -> Parameters
    updateParameters gradients (weights,biases) =
      (
        weights - cmap (*learningRate) regularizedWeightGradients,
        biases - cmap (*learningRate) regularizedBiasGradients
      )
      where
        (regularizedWeightGradients,regularizedBiasGradients) = applyRegularization regularizationTerm gradients

applyRegularization :: Double -> Gradients -> Gradients
applyRegularization regularizationTerm (weightGradients,biasGradients) =
  (
    weightGradients + cmap (*regularizationTerm) weightGradients,
    biasGradients + cmap (*regularizationTerm) biasGradients
  )

runEpochs :: Int -> Double -> Double -> Matrix Double -> Matrix Double -> [Layer] -> [Parameters]
runEpochs numEpochs regularizationTerm learningRate inputs expected = runEpochs' numEpochs
  where
    runEpochs' 0 layers = map layerParameters layers
    runEpochs' remainingEpochs layers = runEpochs' (remainingEpochs - 1) updatedLayers
      where
        gradientMatrices = runBatch inputs expected layers
        parameters = map layerParameters layers
        functions = map activationFunction layers
        updatedLayers = zipWith getLayer updatedParameters functions
        updatedParameters = updateParameters regularizationTerm learningRate gradientMatrices parameters

getLayer :: Parameters -> ActivationFunction -> Layer
getLayer parameters activationFunction = Layer{layerParameters=parameters, activationFunction=activationFunction}
