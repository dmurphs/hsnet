import Numeric.LinearAlgebra
import Test.Hspec

import Lib 
    ( forwardPropogate,
      sigmoid,
      weightsWithInitializer,
      runBatch,
      updateParameters,
      getLayer,
      ActivationFunction (..),
      Activation (..),
      Parameters,
      Gradients,
      Layer (..)
    )

main :: IO ()
main = hspec $ do
  describe "Lib.forwardPropogate" $ do
    it "correctly performs forward propogation" $ do
      let
        activationFunction = ActivationFunction { f=flip (/) 100, f'= \_ -> 1/100 }
        input    = (1><3) [1..]
        layers   = [
                    Layer{layerParameters=((3><4) [1..], fromList [1,1,1,1]),activationFunction=activationFunction},
                    Layer{layerParameters=((4><2) [1..], fromList [2,2]),activationFunction=activationFunction}
                   ]
        expected = [input, (1><4) [0.39,0.45,0.51,0.57], (1><2) [10.28 / 100, 12.2 / 100]] :: [Matrix Double]
        activations = forwardPropogate input layers
      map postActivation activations `shouldBe` expected
  describe "Lib.runBatch" $ do
    it "correctly backpropogates" $ do
      let
        input = (1><1) [1] :: Matrix Double
        expected = (1><1) [1]
        initialWeights = weightsWithInitializer 0.05 [1,1,1]
        initialBiases = map fromList [[0.05], [0.05]] :: [Vector Double]
        zippedInitialParameters = zip initialWeights initialBiases
        activationFunctions = [sigmoid,sigmoid]
        layers = zipWith getLayer zippedInitialParameters activationFunctions
        expectedGradients = [((1><1) [ -1.4970312464482186e-3 ], fromList [-1.4970312464482186e-3]),((1><1) [ -6.303013286946654e-2 ], fromList [-0.12006215555353811])]
        actualGradients = runBatch input expected layers
      actualGradients `shouldBe` expectedGradients
    describe "Lib.updateParameters" $ do
      it "correctly updates parameters" $ do
        let
          learningRate = 0.1
          parameters = [((1><2) [1,2], fromList [1,1])] :: [Parameters]
          gradients = [((1><2) [1,1], fromList [1,1])] :: [Gradients]
          expectedUpdatedParameters = [((1><2) [0.899, 1.899], fromList [0.9,0.9])] :: [Parameters]
          actualUpdatedParameters = updateParameters 0.01 learningRate gradients parameters
        actualUpdatedParameters `shouldBe` expectedUpdatedParameters
