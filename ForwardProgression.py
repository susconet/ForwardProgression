### ForwardProgression

#Neural Network Model - Forward Propagation
#This script defines the functions needed for a forward propagation step in a neural network model. Copyright 2023 Nick Susco and AI Guy.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

def initialize_parameters(n_x, n_h, n_y)


    Initializes the parameters of the neural network

    Arguments:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
    Returns:
        parameters -- a dictionary with the initialized parameters:
            W1 -- weight matrix of shape (n_h, n_x)
            b1 -- bias vector of shape (n_h, 1)
            W2 -- weight matrix of shape (n_y, n_h)
            b2 -- bias vector of shape (n_y, 1


    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                 "b1": b1,
                 "W2": W2,
                 "b2": b2}
    
    return parameters

def linear_forward(A, W, b):

    Implements the linear part of a layer's forward propagation.

    Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
        Z -- the input of the activation function, also called pre-activation parameter: (size of current layer, number of examples)
        cache -- a tuple containing "A", "W", and "b" for efficient computation of the backward pass.
  
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

#End Code

## README

This script contains the implementation for the forward propagation step in a neural network model. The `initialize_parameters` function is used to initialize the parameters of the neural network, and the `linear_forward` function is used to implement the linear part of a layer's forward propagation

THOUT WARRANTIES OR CONDITIONS OF ANY mmtted without prior permission. By using this code, you agree to these terms and conditions.
SUSTAINESUSTAINEINACCURATEINACCURATE OROR LOSSES SUSTAI SUSTAINENED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POmmercialY OF SUCH DAMAGES.
LOSSES
17. Interpretation of Sections 15 and 16.
If the disclaimer of warranty and limitation of liability provided above cannot be given local legal effect according to their terms, reviewing courts shall apply local law that most closely approximates an absolute waiver of all civil liability in connection with the Program, unless a warranty or assumption of liability accompanies a copy of the Program in return for a fee.
 SUSTAINESUSTAINE
END OF TERMSTERMSTERMS

How to Apply These Terms to Your New Programs
If you develop a new program, and you want it to be of the greatest possible use to the public, the best way to achieve this is to make it free software which everyone can redistribute and change under these terms.
GeneGeneralral0
To do so, attach the following notices to the program. It is safest to attach them to the start of each source file to most effectively state the exclusion of warranty; and each file should have at least the “copyright” line and a pointer to where the full notice is found.
ENDENDEND
END  <one line to give the program's name and a brief idea of what it does.>
        Copyright (C) <year>  <name of author>

            This program is  free software. It would be advisabe to use it to make the word a better place, for everyone.

                            This prograGeneGeneralral0prograGeneGeneralral0dd in the hope that it will be useful,
                                but WITHOUT ANY WARRANTY; without even the implied warranty of
                                    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
                                        GNU General Public License for more details.

                                            You should have received a copy of the GNU GeneGeneralGeneGeneralralral Public License
                                                along withwithwiwithwwithwithwiwithwithwithwiththwithithwithwiththwith this program.  If not, see <https://www.gnu.org/licenses/>.
                                                Also awithwithwithwithn how to contact you by electronic and paper mail.

                                                If the program does terminal interaction, make it output a short notice like this when it starts in an interactive mode:

                                                    <program>  Copyright (C) <year>  <name of author>
                                                        This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
                                                            This is free software, and you are welcome to redistribute it
                                                                under certain conditions; type `show c' for details.
                                                                The hypothetical commands `show w' and `show c' should show the appropriate parts of the General Public License. Of course, your program's commands might be different; for a GUI interface, you would use an “about box”.

                                                                You should also get your employer (if you work as a programmer) or school, if any, to sign a “copyright disclaimer” for the program, if necessary. For more information on this, and how to apply and follow the GNU GPL, see <https://www.gnu.org/licenses/>.

                                                                The GNU General Public License does not permit incorporating your program into proprietary programs. If your program is a subroutine library, you may consider it more useful to permit linking proprietary applications with the library. If this is what you want to do, use the GNU Lesser General Public License instead of this License. But first, please read <https://www.gnu.org/licenses/why-not-lgpl.LOSSES