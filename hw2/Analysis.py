# CS122 W'18: Markov models and hash tables
# Keisuke Yokota

import sys
import math
import Hash_Table

HASH_CELLS = 57


class Markov:

    def __init__(self, k, s):
        '''
        Construct a new k-order Markov model using the statistics of 
        string "s"
        '''
        self.character_set = self.get_character_set(s)
        self.hash_table = Hash_Table.Hash_Table(HASH_CELLS, 0)
        self.make_hash(k, s)     
        self.make_hash(k+1, s) 


    # Helper function 1
    def get_character_set(self, s):
        '''
        get a character set in the string speaker
        ''' 
        wordset = set()
        word_list = list(s)
        for i in word_list:
            wordset.add(i)
        return wordset


    # Helper function 2
    def make_hash(self, k, s):
        '''
        split string s by its length k, assign each val and make Hash Table
        '''
        for word in split_str(k, s):
            val = 1
            if self.hash_table.lookup(word):
                val = self.hash_table.lookup(word)[1]+1
            self.hash_table.update(word, val)


    def log_probability(self, s):
        '''
        Get the log probability of string "s", given the statistics of
        character sequences modeled by this particular Markov model
        This probability is *not* normalized by the length of the string.
        '''
        count_S = len(self.character_set)
        if self.hash_table.lookup(s):
            count_M = self.hash_table.lookup(s)[1]
        else:
            count_M = 0
        if self.hash_table.lookup(s[:-1]):
            count_N = self.hash_table.lookup(s[:-1])[1]
        else:
            count_N = 0
        numerator = count_M + 1
        denominator = count_S + count_N
        return math.log(numerator / denominator)


# Helper function 3
def split_str(k, s):
    '''
    split string s by itnteger k
    '''
    length = len(s)
    times = 1 + k // length
    if k % length > 1:
        times += 1
    string = "".join([s] * times) 
    return [string[i:i+k] for i in range(length)]


def identify_speaker(speech1, speech2, speech3, order):
    '''
    Given sample text from two speakers, and text from an unidentified speaker,
    return a tuple with the *normalized* log probabilities of each of 
    the speakers uttering that text under an "order" order character-based 
    Markov model, and a conclusion of which speaker uttered the unidentified
    text based on the two probabilities.
    '''
    speaker1 = Markov(order,speech1)
    speaker2 = Markov(order,speech2)
    unidentified = split_str(order+1, speech3)
    likelihood1 = 0
    likelihood2 = 0
    for word in unidentified:
        likelihood1 += speaker1.log_probability(word)
        likelihood2 += speaker2.log_probability(word)
    conclusion = 'A'
    if likelihood1 < likelihood2:
        conclusion = 'B'
    return (likelihood1/len(speech3), likelihood2/len(speech3), conclusion)


def print_results(res_tuple):
    '''
    Given a tuple from identify_speaker, print formatted results to the screen
    '''
    (likelihood1, likelihood2, conclusion) = res_tuple

    print("Speaker A: " + str(likelihood1))
    print("Speaker B: " + str(likelihood2))

    print("")

    print("Conclusion: Speaker " + conclusion + " is most likely")


if __name__ == "__main__":
    num_args = len(sys.argv)

    if num_args != 5:
        print("usage: python3 " + sys.argv[0] + " <file name for speaker A> " +
              "<file name for speaker B>\n  <file name of text to identify> " +
              "<order>")
        sys.exit(0)

    with open(sys.argv[1], "rU") as file1:
        speech1 = file1.read()

    with open(sys.argv[2], "rU") as file2:
        speech2 = file2.read()

    with open(sys.argv[3], "rU") as file3:
        speech3 = file3.read()

    res_tuple = identify_speaker(speech1, speech2, speech3, int(sys.argv[4]))

    print_results(res_tuple)