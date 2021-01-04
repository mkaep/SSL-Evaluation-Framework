package parser.classifier;

import java.util.Iterator;

import org.deckfour.xes.extension.std.XOrganizationalExtension;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XTrace;

import parser.TraceVariant;

public class ResourceClassifier extends Classifier {
	
	public ResourceClassifier() {
	}
	
	@Override
	public TraceVariant extractTraceVariant(XTrace trace) {
		TraceVariant variant = new TraceVariant();
		Iterator<XEvent> traceIterator = trace.iterator();
		while(traceIterator.hasNext()) {
			XEvent currentEvent = traceIterator.next();
			Event event = new Event(currentEvent.getAttributes().get(XOrganizationalExtension.KEY_RESOURCE).toString());
			variant.addEvent(event);
		}
		return variant;
	}
	

}
