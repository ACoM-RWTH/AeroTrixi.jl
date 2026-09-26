using Documenter
using AeroTrixi

# Set up for local builds
if (get(ENV, "CI", nothing) != "true")
    push!(LOAD_PATH, dirname(@__DIR__))
end

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(AeroTrixi, :DocTestSetup, :(using AeroTrixi); recursive = true)

# Make documentation
makedocs(modules = [AeroTrixi],
         sitename = "AeroTrixi.jl",
         format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                                  canonical = "https://acom-rwth.github.io/AeroTrixi.jl/stable"),
         pages = [
             "Home" => "index.md",
             "Getting started" => "getting_started.md",
             "API Reference" => "api.md"
         ],
         doctest = true,
         linkcheck = false,
         warnonly = [:missing_docs, :cross_references])

# Deploy documentation
deploydocs(repo = "github.com/ACoM-RWTH/AeroTrixi.jl.git",
           devbranch = "main",
           devurl = "dev",
           versions = ["stable" => "v^", "dev" => "dev"],
           push_preview = true)
