package parser.classifier;

import java.util.Iterator;

import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XTrace;

import parser.TraceVariant;

/**
 * Defines the activity classifier. Two traces are considered as identical if the sequence of activities is the same.
 * @author Martin Kaeppel
 */
public class ActivityClassifier extends Classifier {
	
	public ActivityClassifier() {
	}
	
	@Override
	public TraceVariant extractTraceVariant(XTrace trace) {
		TraceVariant variant = new TraceVariant();
		Iterator<XEvent> traceIterator = trace.iterator();
		while(traceIterator.hasNext()) {
			XEvent currentEvent = traceIterator.next();
			Event event = new Event(currentEvent.getAttributes().get(XConceptExtension.KEY_NAME).toString());
			variant.addEvent(event);
		}
		return variant;
	}
}
