# Adding ProofAug Tool to Pure Agentic Prover

## Overview
The pure agentic prover uses a function calling framework where tools are defined in `self.tools` and implemented as methods. Here's how to add your `poofaug` tool.

## 1. Tool Definition
Add your tool to the `self.tools` list (around line 76):

```python
self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "decompose_theorem",
                    "description": "Break a complex theorem into simpler sub-theorems",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "ID of the theorem node to decompose"
                            },
                            "strategy": {
                                "type": "string", 
                                "description": "Decomposition strategy to use",
                                "enum": ["case_analysis", "induction", "contradiction", "construction", "standard"]
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation of why this decomposition strategy is chosen"
                            }
                        },
                        "required": ["node_id", "strategy", "reasoning"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "prove_directly",
                    "description": "Attempt to prove a theorem directly without decomposition",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "ID of the theorem node to prove"
                            },
                            "approach": {
                                "type": "string",
                                "description": "Direct proof approach to try",
                                "enum": ["direct", "contradiction", "induction", "construction", "rewrite"]
                            },
                            "reasoning": {
                                "type": "string", 
                                "description": "Explanation of the proof strategy"
                            }
                        },
                        "required": ["node_id", "approach", "reasoning"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "prove_individual_theorems",
                    "description": "Prove ALL sub-theorems that are children of a decomposed parent theorem. Use the parent node_id, not individual child node_ids.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "parent_node_id": {
                                "type": "string",
                                "description": "ID of the PARENT node whose children should be proved (typically 'root' after decomposition)"
                            },
                            "approach": {
                                "type": "string",
                                "description": "Proof approach to try for each sub-theorem",
                                "enum": ["direct", "contradiction", "induction", "construction", "rewrite", "auto"]
                            }
                        },
                        "required": ["parent_node_id", "approach"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "synthesize_with_lemmas",
                    "description": "Combine proven sub-theorems (as lemmas) into a proof of the parent theorem",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "ID of the parent theorem to synthesize"
                            },
                            "synthesis_strategy": {
                                "type": "string",
                                "description": "How to combine the sub-proofs using lemmas",
                                "enum": ["direct_combination", "case_reunion", "inductive_step", "contradiction_resolution"]
                            }
                        },
                        "required": ["node_id", "synthesis_strategy"]
                    }
                }
            },

            {
                "type": "function",
                "function": {
                    "name": "fix_proof_errors",
                    "description": "Fix errors in a failed proof by sending Lean error feedback back to Kimina for correction",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "ID of the node with failed proof to fix"
                            },
                            "max_attempts": {
                                "type": "integer",
                                "description": "Maximum number of error fix attempts",
                                "default": 3
                            }
                        },
                        "required": ["node_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "verify_proof",
                    "description": "Verify a proof using the Lean 4 verifier",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string", 
                                "description": "ID of the node with proof to verify"
                            }
                        },
                        "required": ["node_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_proof_state",
                    "description": "Get current state of the proof tree and progress",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "detail_level": {
                                "type": "string",
                                "description": "Level of detail to return",
                                "enum": ["summary", "detailed", "full"]
                            }
                        },
                        "required": ["detail_level"]
                    }
                }
            }
        ]
```

## 2. Tool Implementation
Here's the concrete example of `_fix_proof_errors` to follow:

```python
def _fix_proof_errors(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Fix errors in a failed proof using Lean error feedback and Kimina (following error_fix.py pattern)"""
        requested_node_id = args["node_id"]
        max_attempts = args.get("max_attempts", 3)
        
        # Resolve node ID
        node_id = self._resolve_node_id(requested_node_id)
        if node_id is None:
            return {"status": "error", "message": f"Node {requested_node_id} not found"}
        
        node = self.nodes[node_id]
        
        if not node.proof_text:
            return {
                "status": "error", 
                "message": f"Node {node_id} has no proof to fix. Use prove_directly first to generate an initial proof attempt.",
                "suggestion": f"Try: prove_directly(node_id='{node_id}') first, then fix_proof_errors"
            }
        
        if node.status == "proved":
            return {"status": "completed", "message": "Proof is already valid - no fixing needed"}
        
        print(f"🛠️  FIXING PROOF ERRORS: {node_id} (max {max_attempts} attempts)")
        self.strategies_used.append("error_fixing")
        
        current_proof = node.proof_text
        
        # Check if proof_text contains raw LLM response instead of extracted proof
        if not current_proof.strip().startswith(("import", "theorem", "lemma", "def")):
            print(f"   🔍 Proof text appears to be raw LLM response, attempting extraction...")
            extracted = extract_proof_from_text(current_proof)
            if extracted and len(extracted.strip()) > 10:
                current_proof = extracted
                node.proof_text = current_proof
                print(f"   ✅ Extracted proof from raw response")
            else:
                return {
                    "status": "error",
                    "message": f"Cannot extract valid proof from stored response. Raw response too malformed.",
                    "suggestion": f"Try prove_directly(node_id='{node_id}') again with different approach"
                }
        
        fix_attempts = 0
        
        for attempt in range(max_attempts):
            fix_attempts += 1
            print(f"\n🔧 Error fix attempt {fix_attempts}/{max_attempts}")
            
            # Step 1: Verify current proof to get detailed Lean feedback
            if not self.lean_verifier:
                return {
                    "status": "error", 
                    "message": "Lean verifier not available - cannot perform error fixing",
                    "suggestion": "Use simulation mode or configure Lean verifier"
                }
            
            try:
                print(f"   🔍 Getting Lean error feedback...")
                verification_result = self.lean_verifier.verify(current_proof)
                
                # Check if proof is now valid
                if verification_result.get("is_valid_no_sorry", False):
                    print(f"   ✅ Proof is now valid after {fix_attempts} attempts!")
                    node.status = "proved"
                    node.proof_text = current_proof
                    node.error_message = None
                    
                    return {
                        "status": "completed",
                        "result": "fixed",
                        "attempts_used": fix_attempts,
                        "final_proof": current_proof,
                        "verification": verification_result,
                        "message": f"Successfully fixed proof after {fix_attempts} attempts"
                    }
                
                # Get detailed error feedback
                lean_feedback = verification_result
                if not lean_feedback.get("has_error", True):
                    # No errors but still not valid_no_sorry - might have sorry
                    error_msg = "Proof contains 'sorry' or other issues preventing validation"
                else:
                    error_msg = f"Proof has {len(lean_feedback.get('diagnostics', []))} error(s)"
                
                print(f"   ❌ {error_msg}")
                
                # Step 2: Format error feedback for Kimina (following error_fix.py pattern)
                formatted_feedback = create_tool_message(
                    formal_code=current_proof,
                    lean_feedback=lean_feedback
                )
                
                print(f"   📋 Sending error feedback to Kimina ({len(formatted_feedback)} chars)")
                
                # Step 3: Ask Kimina to fix the errors
                messages = [
                    {"role": "system", "content": "You are an expert programmer and mathematician who helps formalizing mathematical problems in Lean 4."},
                    {"role": "user", "content": f"The following Lean 4 proof has errors. Please fix them:\n\n{node.statement}"},
                    {"role": "assistant", "content": f"Here's my proof:\n\n{current_proof}"},
                    {"role": "user", "content": formatted_feedback}
                ]
                
                response = self.llm_services.prove_client.chat.completions.create(
                    model=self.llm_services.prove_model_name,
                    messages=messages,
                    temperature=0.6,  # Slightly higher temp for error fixing creativity
                    max_tokens=8192,
                    n=1,
                )
                
                fix_response = response.choices[0].message.content.strip()
                print(f"   📝 Kimina fix response: {len(fix_response)} chars")
                
                # Step 4: Extract the fixed proof
                fixed_proof = extract_proof_from_text(fix_response)
                
                if not fixed_proof or len(fixed_proof) < 20:
                    print(f"   ⚠️  Could not extract valid fixed proof from response")
                    continue
                
                # Make sure it's for the right theorem
                current_proof = replace_statement_in_proof(fixed_proof, node.statement)
                print(f"   🔍 Extracted fixed proof: {len(current_proof)} chars")
                
            except Exception as e:
                print(f"   ❌ Error in fix attempt {fix_attempts}: {e}")
                continue
        
        # All attempts failed
        print(f"\n❌ Error fixing FAILED after {max_attempts} attempts")
        node.status = "failed"
        node.error_message = f"Error fixing failed after {max_attempts} attempts"
        
        return {
            "status": "completed",
            "result": "failed",
            "attempts_used": fix_attempts,
            "max_attempts": max_attempts,
            "message": f"Could not fix proof errors after {max_attempts} attempts",
            "suggestion": "Try different proof approach or manual debugging",
            "final_proof": current_proof
        }
```

## 3. Function Router
Add your tool to the `_execute_function` method routing:

```python
elif function_name == "poofaug":
    return self._poofaug(arguments)
```

## Key Implementation Points

### Node Management
- **Node Resolution**: Always use `self._resolve_node_id()` 
- **Node Access**: Use `node.statement`, `node.proof_text`, `node.status`
- **Status Updates**: Set `node.status = "proved"/"failed"` as appropriate

### Error Handling
- Return `{"status": "error"}` for invalid inputs
- Include helpful `"suggestion"` fields for user guidance

### System Integration
- **Strategy Tracking**: Add `self.strategies_used.append("poofaug")`
- **Lean Verification**: Access via `self.lean_verifier.verify()` if needed
- **Utilities**: Use existing helpers like `extract_proof_from_text()`

## Return Format
Always return a dictionary with:
- `"status"`: `"completed"`, `"error"`, or `"failed"`
- `"message"`: Human readable result
- Additional fields specific to your tool

## Implementation Template

```python
def _poofaug(self, args: Dict[str, Any]) -> Dict[str, Any]:
    """Your ProofAug implementation"""
    requested_node_id = args["node_id"]
    
    # Resolve node ID with fallback logic
    node_id = self._resolve_node_id(requested_node_id)
    if node_id is None:
        return {"status": "error", "message": f"Node {requested_node_id} not found"}
    
    node = self.nodes[node_id]
    self.strategies_used.append("poofaug")
    
    # Your ProofAug logic here
    
    return {
        "status": "completed",
        "result": "your_result",
        # Add your return fields
    }
```

---

The agent will automatically discover and use your tool via function calling once integrated. 