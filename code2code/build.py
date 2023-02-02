from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'my-languages.so',

  # Include one or more languages
  [
    'tree-sitter-java',
    'tree-sitter-c-sharp'
  ]
)

