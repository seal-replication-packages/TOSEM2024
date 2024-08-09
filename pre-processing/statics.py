other_metrics = ['has_py_file', 'lines_added', 'lines_deleted', 'lines_changed', 'files', 'code_entropy',
                    'words_count', 'sentences_count', 'readability',
                    'refactoring_contribution_ratio',]

code_metrics = ['AvgCountLine', 'AvgCountLineBlank', 'AvgCountLineCode', 'AvgCountLineComment', 
                'AvgCyclomatic', 'CountClassBase', 'CountClassCoupled', 'CountClassCoupledModified', 
                'CountClassDerived', 'CountDeclClass', 'CountDeclExecutableUnit', 'CountDeclFile', 
                'CountDeclFunction', 'CountDeclInstanceMethod', 'CountDeclInstanceVariable', 
                'CountDeclMethod', 'CountDeclMethodAll', 'CountLine', 'CountLineBlank', 'CountLineCode', 
                'CountLineCodeDecl', 'CountLineCodeExe', 'CountLineComment', 'CountStmt', 'CountStmtDecl', 
                'CountStmtExe', 'Cyclomatic', 'MaxCyclomatic', 'MaxInheritanceTree', 'MaxNesting', 'RatioCommentToCode', 'SumCyclomatic']

refined_metrics = ['lines_added', 'lines_deleted', 'lines_changed', 'code_entropy',
                'words_count', 'sentences_count', 'readability',
                'refactoring_contribution_ratio', 'AvgCountLine', 'AvgCountLineBlank', 'AvgCountLineCode', 'AvgCountLineComment', 
                'AvgCyclomatic', 'CountClassCoupled', 'CountClassCoupledModified', 
                'CountClassDerived', 'CountDeclClass', 'CountDeclFile', 
                'CountDeclFunction', 'CountDeclInstanceMethod', 'CountDeclInstanceVariable', 
                'CountDeclMethod', 'CountDeclMethodAll', 'CountLineBlank', 'CountLineComment', 
                'Cyclomatic', 'MaxCyclomatic', 'MaxInheritanceTree', 'MaxNesting', 'RatioCommentToCode', 'SumCyclomatic']

process_metrics = other_metrics+code_metrics
