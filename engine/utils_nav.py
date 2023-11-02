import os
from PIL import Image
import openai
import numpy as np
import copy
import re

from .step_interpreters_nav import register_step_interpreters, parse_step

class Program:
    def __init__(self,prog_str,init_state=None):
        self.prog_str = prog_str
        self.state = init_state if init_state is not None else dict()
        # self.instructions = self.prog_str.split('\n')
        self.instructions_dict = self.clear_splits(prog_str)

    def clear_splits(self, prog_str):
        instructions = re.findall(r'while not \w+:\n\s*(.*?)(?=\nwhile not|\Z)', prog_str, re.DOTALL)

        instructions_dict = {}
        for split_counter, instr in enumerate(instructions, start=1):
            cleaned_instruction = instr.rstrip('\n').replace('\\', '')
            cleaned_lines = [line.strip() for line in cleaned_instruction.split('\n')]
            instructions_dict[f'split{split_counter}'] = cleaned_lines
        return instructions_dict


class ProgramInterpreter:
    def __init__(self,dataset='nlvr'):
        self.step_interpreters = register_step_interpreters(dataset)

    def execute_step(self,prog_step,inspect):
        step_name = parse_step(prog_step.prog_str,partial=True)['step_name']
        print(step_name)
        return self.step_interpreters[step_name].execute(prog_step,inspect)
    
    def execute_loop(self,prog, prog_steps,inspect):

        end_episode, trials = False, 1
        html_tmp = '<hr>'
        while not end_episode and trials < 5:
            print('-------------')
            print(f'| Trial n°{trials} |')
            print('-------------')
            
            for prog_step in prog_steps:
                if inspect:
                    step_output, step_html = self.execute_step(prog_step,inspect)
                    html_tmp += step_html + '<hr>'
                else:
                    step_output = self.execute_step(prog_step,inspect)

                # check if stop is called, else loop (navigate)
                if isinstance(step_output, str):
                    if step_output in ['STOP','NAVIGATE']:
                        end_episode = step_output == 'STOP'
            
            # take into account max 3 trials for each NAV episode
            trials += 1 if not end_episode else 1

        return step_output, prog.state, html_tmp

    def execute(self,prog,init_state,inspect=False):
        if isinstance(prog,str):
            prog = Program(prog,init_state)
            prog.state = dict(init_state) if init_state is not None else dict()
        else:
            assert(isinstance(prog,Program))

        html_str = '<hr>'
        for splits in prog.instructions_dict.keys():
            prog_steps = [Program(instruction,init_state=prog.state) \
                        for instruction in prog.instructions_dict[splits]]

            step_output, prog.state, html_split = self.execute_loop(prog, prog_steps, inspect)
            html_str+=html_split

        return step_output, prog.state, html_str
    

class ProgramGenerator():
    def __init__(self,prompter,temperature=0.7,top_p=0.5,prob_agg='mean', debug=False):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.prompter = prompter
        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg
        self.debug = debug

    def compute_prob(self,response):
        eos = '<|endoftext|>'
        for i,token in enumerate(response.choices[0]['logprobs']['tokens']):
            if token==eos:
                break

        if self.prob_agg=='mean':
            agg_fn = np.mean
        elif self.prob_agg=='sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0]['logprobs']['token_logprobs'][:i]))

    def generate(self,inputs, debug_answer):
        if self.debug:
            return debug_answer.lstrip('\n').rstrip('\n'), None
        
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=self.prompter(inputs),
            temperature=self.temperature,
            max_tokens=512,
            top_p=self.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
            logprobs=1
        )

        prob = self.compute_prob(response)
        prog = response.choices[0]['text'].lstrip('\n').rstrip('\n')
        return prog, prob
    