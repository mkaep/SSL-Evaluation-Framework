package parser.classifier;

import java.util.Iterator;

import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.extension.std.XLifecycleExtension;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XTrace;

import parser.TraceVariant;

public class ActivityLifecycleClassifier extends Classifier {
	
	public ActivityLifecycleClassifier() {
		
	}

	@Override
	public TraceVariant extractTraceVariant(XTrace trace) {
		TraceVariant variant = new TraceVariant();
		Iterator<XEvent> traceIterator = trace.iterator();
		while(traceIterator.hasNext()) {
			XEvent currentEvent = traceIterator.next();
			String activity = currentEvent.getAttributes().get(XConceptExtension.KEY_NAME).toString();
			String lifecycle = currentEvent.getAttributes().get(XLifecycleExtension.KEY_TRANSITION).toString();

			variant.addEvent(new Event(activity+"-"+lifecycle));
		}
		return variant;
	}

}
