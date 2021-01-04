package parser.classifier;

import org.deckfour.xes.model.XTrace;

import parser.TraceVariant;

/**
 * Abstract class to define further classifier
 * @author Martin Kaeppel
 */
public abstract class Classifier {
	public abstract TraceVariant extractTraceVariant(XTrace trace); 
}
