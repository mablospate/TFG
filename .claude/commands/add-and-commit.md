Add and commit every change that happens in this session.

Add a descriptive message about all changes that have been included, prioritizing bullet lists with straight information.

Additionally, add one of the following instructions at the beginning of the message depending on the changes:

- **feat**: (new feature for the user, not a new feature for build script)
- **fix**: (bug fix for the user, not a fix to a build script)
- **docs**: (changes to the documentation)
- **style**: (formatting, missing semi colons, etc; no production code change)
- **refactor**: (refactoring production code, eg. renaming a variable)
- **test**: (adding missing tests, refactoring tests; no production code change)
- **chore**: (Update of build tasks, package admin configuration; No changes to the code.)

Finally, add the key branch for commit tracking in issues created in **JIRA**. This key can be found at the beginning of the branch name (e.g. *feature/{key-name}-{branch-name}* or *{key-name}-{branch-name}*). This information can be found using `git branch --show-current`. Add it into the first part of the message following this structure: *{instruction-name} ({key-name}): {message}*

## Example
feat (EEAI-2): add toggle functionality
fix (EEAI-3): remove unused button
docs (EEAI-4): add new documentation about RAG system
