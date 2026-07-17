# Maintenance commands for rishav1.github.io (al-folio v1, native Ruby — no Docker).
#
# Prereqs (one-time):
#   brew install ruby imagemagick
#   make install
#
# Daily:
#   make serve     # live-reload dev server at http://localhost:4000
#   make build     # production build into _site/
#   make resume    # recompile resume/resume.tex -> assets/pdf/resume.pdf (needs LaTeX)
#   make clean     # remove build output
#   make deploy    # push master -> origin (GitHub Actions publishes)

# Use the Homebrew Ruby (keg-only) without requiring a global switch.
RUBY_BIN := /opt/homebrew/opt/ruby/bin
SHELL    := /bin/bash
PORT     ?= 4000
export PATH := $(RUBY_BIN):$(PATH)

.PHONY: help install serve build clean resume deploy

help:
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?# "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: # install Ruby gems (bundler)
	bundle install

serve: # live-reload dev server at http://localhost:$(PORT) (local links stay local)
	bundle exec jekyll serve --livereload --port $(PORT) --config _config.yml,_config.dev.yml --trace

build: # production build into _site/
	bundle exec jekyll build

clean: # remove build output and caches
	bundle exec jekyll clean

resume: # recompile the LaTeX resume and refresh the linked PDF
	cd resume && latexmk -pdf -interaction=nonstopmode -halt-on-error resume.tex && latexmk -c
	cp resume/resume.pdf assets/pdf/resume.pdf
	@echo "Updated assets/pdf/resume.pdf"

deploy: # push master to origin (triggers the GitHub Actions Pages deploy)
	git push origin master
