import re
import pandas as pd

class Preprocessing:
    """
    Preprocess question-answer (q,a) pairs with various strategies:

    """

    @staticmethod
    def process_data(sample, boxed = False):
        """
        Filter the answer to only contain the last \booxed{...} or \fbox{...}

        Parameters:
        sample: Tuple[str, str] - (problem, solution) pair // From the tuple only problem-solution pair will be extracted

        Returns:
        Tuple[str, str] or None - Filtered (problem, solution) or None if not found
        """
        q = sample['problem']
        a = sample['solution']
        if boxed:
            ca = a.map(Preprocessing.extract_boxed_answer)
            ca = ca.map(Preprocessing.clean_numbers)
            
        else:
            ca = a.map(Preprocessing.clean_numbers)

        if ca is None:
            return None
        
        return pd.DataFrame({"problem": q, "solution": ca})
    
    @staticmethod
    def extract_boxed_answer(string: str) -> str | None:
        if string is None:
            return None
        
        start = string.find(r'\boxed{')
        if start == -1:
            start = string.rfind(r'\fbox{')
            if start == -1:
                return None
            
        
        brace_start = string.find('{', start)
        if brace_start == -1:
            return None

        brace_count = 0
        end = -1
        for i in range(brace_start, len(string)):
            if string[i] == '{':
                brace_count += 1
            elif string[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break

        if end == -1:
            return None 
    
        extracted_value = string[brace_start + 1:end].strip()

        if extracted_value.startswith('{') and extracted_value.endswith('}'):
            extracted_value = extracted_value[1:-1]
        extracted_value = extracted_value.replace(' ', '') 
        print("Extracted_value : ", extracted_value)


        return extracted_value


    @staticmethod
    def refine_box_extraction(string: str):
                
            if string is None:
                return None
            boxed_pos = string.find(r'\boxed')
            if boxed_pos == -1:
                return None

            start_pos = boxed_pos + len(r'\boxed')
            trim_string = string[start_pos:]

            dollar_pos = trim_string.find(r'$')
            if dollar_pos == -1:
                return None
            end_pos = start_pos + dollar_pos 

            extracted_value = string[start_pos:end_pos]
            print("extracted Value : : ", extracted_value)
            return extracted_value if extracted_value else None
    
    @staticmethod
    def clean_numbers(string):
        """
        Add comma to numbers longer than 3 digits within string.
        Example: 'I got 123456 marks' => 'I got 123,456 marks'

        Returns:
        string: str

        Usage:
        ----------------------
        text = "going to collect 1000000 from somebody."
        prep_data = Preprocessing.clean_numbers(text)

        Output:
        "going to collect 1,000,000 from somebody."
        ----------------------
        """
        if not string:
            return None
        
        num_prev_digits = 0
        new_string = ""
        for i, c in enumerate(string):
            if c in '0123456789':
                num_prev_digits += 1
            else:
                if num_prev_digits > 3:
                    string_number = new_string[-num_prev_digits:]
                    new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
                num_prev_digits = 0
            new_string += c

        if num_prev_digits > 3:
            string_number = new_string[-num_prev_digits:]
            new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

        return new_string
    
    